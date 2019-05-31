import numpy as np
import tensorflow as tf
import keras
from keras.layers import *
from keras.layers.merge import _Merge
from keras.activations import relu
from keras.models import Model, Sequential
from keras.models import load_model
import keras.backend as Ki
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
import time
import sys
sys.path.append('./utils')
sys.path.append('./networks')
import logging
import logging_utils
from parameters import load_params
import plots
import tboard
from SpecNormLayers import ConvSN2D, ConvSN2DTranspose, DenseSN
from functools import partial

_SNlayertypes = [type(ConvSN2D(filters=2, kernel_size=2)), type(ConvSN2DTranspose(5, 5)), type(DenseSN(1))]


class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


class WGAN_GP:
    
    def __init__(self, configtag, expDir):

        # Load hyperparmeters
        self.configtag = configtag
        logging.info('Parameters:')
        self.init_params(configtag)
        self.expDir = expDir
        self.inits = {'dense':keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
                      'conv':keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02),
                      'tconv':keras.initializers.RandomNormal(mean=0.0, stddev=0.02)}

        # Import slices
        self.real_imgs = np.load('./data/'+self.dataname+'_train.npy')
        self.val_imgs = np.load('./data/'+self.valname+'_val.npy')
        self.n_imgs = self.real_imgs.shape[0]
        if self.datafmt == 'channels_last':
            self.real_imgs = np.moveaxis(self.real_imgs, 1, -1)
            self.val_imgs = np.moveaxis(self.val_imgs, 1, -1)
        
        # Build networks
        self.discrim, self.genrtor = self.load_networks()

        # Build critic
        self.genrtor.trainable = False
        real_img = Input(shape=self.imshape)
        z_disc = Input(shape=(1,self.noise_vect_len))
        fake_img = self.genrtor(z_disc)
        interp_img = RandomWeightedAverage()([real_img, fake_img])

        fake = self.discrim(fake_img)
        real = self.discrim(real_img)
        interp = self.discrim(interp_img)

        partial_gp_loss = partial(self.gradient_penalty_loss, averaged_samples=interp_img)
        partial_gp_loss.__name__ = 'gradient_penalty'
        self.critic = Model(inputs=[real_img, z_disc], outputs=[real, fake, interp])
        self.critic.compile(loss=[self.mean_loss, self.mean_loss, partial_gp_loss],
                            optimizer=keras.optimizers.Adam(lr=self.D_lr, beta_1=0.5),
                            loss_weights=[1,1,10])

        # Build generator
        self.discrim.trainable = False
        self.genrtor.trainable = True
        z = Input(shape=(1,self.noise_vect_len))
        genimg = self.genrtor(z)
        decision = self.discrim(genimg)
        self.stacked = Model(z, decision)
        self.stacked.compile(loss=self.mean_loss, optimizer=keras.optimizers.Adam(lr=self.G_lr, beta_1=0.5))

        # Setup tensorboard stuff
        self.TB_genimg = tboard.TboardImg('genimg')
        self.TB_pixhist = tboard.TboardImg('pixhist')
        self.TB_scalars = tboard.TboardScalars()
        if self.weight_hists:
            self.TB_Dwts = tboard.TboardHists('D')
            self.TB_Gwts = tboard.TboardHists('G')
        if self.grad_hists:
            self.TB_Ggrads = tboard.TboardHists('Ggrad')
        if self.specnorm and self.sigma_plot:
            self.TB_Dsigmas = tboard.TboardSigmas('Discriminator')
            self.TB_Gsigmas = tboard.TboardSigmas('Generator')


    def load_networks(self):
        if self.resume:
            logging.info("Resuming D: %s, G: %s"%(self.resume['D'], self.resume['G']))
            custom_lyrs = {'DenseSN':DenseSN, 'ConvSN2D':ConvSN2D, 'ConvSN2DTranspose':ConvSN2DTranspose}
            return ( load_model(self.resume['D'], custom_objects=custom_lyrs),
                     load_model(self.resume['G'], custom_objects=custom_lyrs) )
        else:
            return ( self.build_discriminator(), self.build_generator() )


    def mean_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)

    def build_discriminator(self):
        
        dmodel = Sequential()
        for lyrIdx in range(self.nlayers):
            if self.doubleconv:
                dmodel.add(self._conv_layer_type(1, lyrIdx))
            dmodel.add(self._conv_layer_type(self.convstride, lyrIdx)) 
            #dmodel.add(BatchNormalization(epsilon=1e-5, momentum=0.9, axis=self.bn_axis))
            dmodel.add(LeakyReLU(alpha=self.alpha))
        dmodel.add(Flatten(data_format=self.datafmt))
        dmodel.add(self._dense_layer_type(1, shape_spec=False))
        if self.loss == 'hinge': # hinge loss is untested
            act = 'tanh'
        else:
            act = 'sigmoid'
        #dmodel.add(Activation(act)) # remove sigmoid activation to avoid bug w/ default Keras loss
        dmodel.summary(print_fn=logging_utils.print_func)
        img = Input(shape=self.imshape)
        return Model(img, dmodel(img))


    def build_generator(self):
        fmapsize = self.img_dim//int(2**self.nlayers)
        gmodel = Sequential()
        gmodel.add(self._dense_layer_type(fmapsize*fmapsize*(2*self.ndeconvfilters[0]), shape_spec=(1,self.noise_vect_len)))
        gmodel.add(BatchNormalization(epsilon=1e-5, momentum=0.9, axis=self.bn_axis))
        gmodel.add(ReLU())
        if self.datafmt == 'channels_last':
            fmapshape = (fmapsize, fmapsize, 2*self.ndeconvfilters[0])
        else:
            fmapshape = (2*self.ndeconvfilters[0], fmapsize, fmapsize)
        gmodel.add(Reshape(fmapshape))
        for lyrIdx in range(self.nlayers):
            if self.doubleconv:
                gmodel.add(self._deconv_layer_type(1, lyrIdx))
            gmodel.add(self._deconv_layer_type(self.convstride, lyrIdx)) 
            if lyrIdx == self.nlayers - 1:
                gmodel.add(Activation('tanh')) # last layer is deconv+tanh
            else:
                gmodel.add(BatchNormalization(epsilon=1e-5, momentum=0.9, axis=self.bn_axis)) # hidden layers are deconv+batchnorm+relu
                gmodel.add(ReLU())
        gmodel.summary(print_fn=logging_utils.print_func)
        noise = Input(shape=(1,self.noise_vect_len))        
        return Model(noise, gmodel(noise))


    def _dense_layer_type(self, num, shape_spec=False):

        lyr = None
        if shape_spec:
            if self.specnorm:
                # Use DenseSN if you want spectral normalization in G
                lyr = Dense(num, input_shape=shape_spec, kernel_initializer=self.inits['dense'])
            else:
                lyr = Dense(num, input_shape=shape_spec, kernel_initializer=self.inits['dense'])
        else:
            if self.specnorm:
                # Use DenseSN if you want spectral normalization in D
                lyr = DenseSN(num, kernel_initializer=self.inits['dense'])
            else:
                lyr = Dense(num, kernel_initializer=self.inits['dense'])
        return lyr


    def _conv_layer_type(self, stride, lyrIdx):

        lyr = None
        if lyrIdx == 0:
            if self.specnorm:
                # Use ConvSN2D if you want spectral normalization in D
                lyr = ConvSN2D(filters=self.nconvfilters[lyrIdx], kernel_size=self.convkern, strides=stride, 
                             padding='same', input_shape=self.imshape, data_format=self.datafmt, 
                             kernel_initializer=self.inits['conv'])
            else:
                lyr = Conv2D(filters=self.nconvfilters[lyrIdx], kernel_size=self.convkern, strides=stride, 
                             padding='same', input_shape=self.imshape, data_format=self.datafmt,
                             kernel_initializer=self.inits['conv'])
        else:
            if self.specnorm:
                # Use ConvSN2D if you want spectral normalization in D
                lyr = ConvSN2D(filters=self.nconvfilters[lyrIdx], kernel_size=self.convkern, strides=stride,
                             padding='same', data_format=self.datafmt, kernel_initializer=self.inits['conv'])
            else:
                lyr = Conv2D(filters=self.nconvfilters[lyrIdx], kernel_size=self.convkern, strides=stride,
                             padding='same', data_format=self.datafmt, kernel_initializer=self.inits['conv'])
        return lyr



    def _deconv_layer_type(self, stride, lyrIdx):

        lyr = None
        if self.specnorm:
            # Use ConvSN2DTranspose if you want spectral normalization in G
            lyr = Conv2DTranspose(self.ndeconvfilters[lyrIdx], self.convkern, strides=stride,
                                    padding='same', data_format=self.datafmt, kernel_initializer=self.inits['tconv'])
        else:
            lyr = Conv2DTranspose(self.ndeconvfilters[lyrIdx], self.convkern, strides=stride,
                                  padding='same', data_format=self.datafmt, kernel_initializer=self.inits['tconv'])
        return lyr



    def init_params(self,configtag):
        
        params = load_params('./config.yaml', configtag)
        logging.info(str(params))
        self.dataname = params['dataname']
        self.valname = params['valname']
        self.img_dim = params['img_dim']
        self.noise_vect_len = params['noise_vect_len']
        self.nlayers = params['nlayers']
        self.convkern = params['convkern']
        self.convstride = params['convstride']
        self.nconvfilters = params['nconvfilters']
        self.ndeconvfilters = params['ndeconvfilters']
        self.specnorm = params['specnorm']
        self.label_flip = params['label_flip']
        self.batchsize = params['batchsize'] 
        self.print_batch = params['print_batch']
        self.checkpt_batch = params['checkpt_batch']
        self.cscale = params['cscale']
        self.datascale = params['datascale']
        self.G_lr, self.D_lr = params['learn_rate']
        self.DG_update_ratio = params['DG_update_ratio']
        self.resume = params['resume']
        self.Nepochs = params['Nepochs']
        self.loss = params['loss']
        self.weight_hists = params['weight_hists']
        self.grad_hists = params['grad_hists']
        self.sigma_plot = params['sigma_plot']
        self.doubleconv = params['doubleconv']
        self.datafmt = params['datafmt']
        self.alpha = 0.2
        self.start = 0
        self.bestchi = np.inf
        if self.datafmt == 'channels_last':
            self.bn_axis = -1
            self.imshape = (self.img_dim, self.img_dim, 1)
        else:
            self.bn_axis = 1
            self.imshape = (1, self.img_dim, self.img_dim)
        self.real = -1
        self.fake = 1

    def train_epoch(self, shuffler, num_batches, epochIdx):
        
        d_losses = []
        d_real_losses = []
        d_fake_losses = []
        g_losses = []

        for batch in range(num_batches):
            iternum = (epochIdx*num_batches + batch)
            t1 = time.time()

            reals = self.real*np.ones((self.batchsize,1))
            fakes = self.fake*np.ones((self.batchsize,1))
            dontcare = np.ones((self.batchsize,1)) # dummy labels for gradient penalty loss
            labelflip = self.label_flip
            for i in range(reals.shape[0]):
                if np.random.uniform(low=0., high=1.0) < labelflip:
                    reals[i,0] = self.fake
                    fakes[i,0] = self.real

            # train critic
            for iters in range(self.DG_update_ratio//2):
                start_idx = batch*self.batchsize*(self.DG_update_ratio//2) + iters*self.batchsize
                end_idx = start_idx + self.batchsize
                real_img_batch = self.real_imgs[shuffler[start_idx:end_idx]]
                noise_vects = np.random.normal(loc=0.0, size=(self.batchsize, 1, self.noise_vect_len))

                discr_all_losses = self.critic.train_on_batch([real_img_batch, noise_vects],
                                                             [reals, fakes, dontcare])
                discr_loss = discr_all_losses[0]
                d_losses.append(discr_loss)

            # train generator via stacked model
            genrtr_loss = self.stacked.train_on_batch(noise_vects, self.real*np.ones((self.batchsize,1)))
            t2 = time.time()

            g_losses.append(genrtr_loss)

            
            t2 = time.time()

            if batch%self.print_batch == 0:
                logging.info("| --- batch %d of %d --- |"%(batch + 1, num_batches))
                logging.info("|Discriminator: loss=%f"%(discr_loss))
                logging.info("|Generator: loss=%f"%(genrtr_loss))
                logging.info("|Time: %f"%(t2-t1))
            if iternum%self.checkpt_batch == 0:
                # Tensorboard monitoring
                iternum = iternum/self.checkpt_batch
                self.TB_genimg.on_epoch_end(iternum, self, logs={})
                chisq = self.TB_pixhist.on_epoch_end(iternum, self, logs={})
                scalars = {'d_loss':np.mean(d_losses), 'd_real_loss':np.mean(d_real_losses),
                           'd_fake_loss':np.mean(d_fake_losses), 'g_loss':np.mean(g_losses), 'chisq':chisq}
                self.TB_scalars.on_epoch_end(self, iternum, scalars)

                sess = K.get_session()
                
                if self.weight_hists:
                    # Monitor histogram of weights for conv and dense layers
                    Dlayerdict = {layer.name:layer.get_weights()[0] for layer in self.discrim.layers[1].layers \
                                                                    if 'conv' in layer.name or 'dense' in layer.name}
                    Glayerdict = {layer.name:layer.get_weights()[0] for layer in self.genrtor.layers[1].layers \
                                                                    if 'conv' in layer.name or 'dense' in layer.name}

                    self.TB_Dwts.on_epoch_end(self, iternum, Dlayerdict)
                    self.TB_Gwts.on_epoch_end(self, iternum, Glayerdict)

                if self.grad_hists:
                    # Monitor histogram of gradients -- VERY slow and crashes when using big network
                    ggrad_tensors = self._get_grad_tensors(self.stacked, tf.ones((self.batchsize, 1)), \
                                                           tf.random.normal((self.batchsize, 1, self.noise_vect_len)), stacked=True)
                    ggrads = sess.run(ggrad_tensors)
                    self.TB_Ggrads.on_epoch_end(self, iternum, ggrads)
                if self.specnorm and self.sigma_plot:
                    # Monitor spectral norm of normalized weight matrices
                    dsigmas = [layer.sigma.eval(session=sess)[0,0] for layer in self.discrim.layers[1].layers\
                                                                   if type(layer) in _SNlayertypes]
                    gsigmas = [layer.sigma.eval(session=sess)[0,0] for layer in self.genrtor.layers[1].layers\
                                                                   if type(layer) in _SNlayertypes]
                    
                    dsiglabels = ['conv'+str(idx+1) for idx in range(self.nlayers)] + ['dense_d']
                    gsiglabels = ['dense_g'] + ['tconv'+str(idx+1) for idx in range(self.nlayers)]
                    
                    dsiglog = dict(zip(dsiglabels, dsigmas))
                    gsiglog = dict(zip(gsiglabels, gsigmas))
                    self.TB_Dsigmas.on_epoch_end(self, iternum, dsiglog)
                    self.TB_Gsigmas.on_epoch_end(self, iternum, gsiglog)
                
                d_losses = []
                d_real_losses = []
                d_fake_losses = []
                g_losses = []
                
                if chisq<self.bestchi and iternum>50:
                    # update best chi-square score and save
                    self.bestchi = chisq
                    self.genrtor.save(self.expDir+'models/g_cosmo_best.h5')
                    self.discrim.save(self.expDir+'models/d_cosmo_best.h5')
                    logging.info("BEST saved at %d, chi=%f"%(iternum, chi))


    def _get_grad_tensors(self, model, labels, data, stacked=False):
        
        loss = keras.losses.get(self.loss)
        if stacked:
            learn_rate = self.G_lr
            weights = [[layer.name, layer.trainable_weights[0]] for layer in model.layers[1].layers[1].layers \
                                                                if 'conv' in layer.name or 'dense' in layer.name]
        else:
            learn_rate = self.D_lr
            weights = [[layer.name, layer.trainable_weights[0]] for layer in model.layers[1].layers \
                                                                if 'conv' in layer.name or 'dense' in layer.name]
        optim = keras.optimizers.Adam(learn_rate, beta_1 = 0.5)
        grads = optim.get_gradients(loss(labels, model(data)), [wvars for _ , wvars in weights])
        gradsdict = dict(zip([name for name, _ in weights], grads))
        return gradsdict








