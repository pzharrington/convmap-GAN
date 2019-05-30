import numpy as np
import tensorflow as tf
import keras
from keras.layers import *
from keras.activations import relu
from keras.models import Model, Sequential
from keras.models import load_model
import keras.backend as K
import time
import sys
sys.path.append('./utils')
sys.path.append('./networks')
import logging
import logging_utils
from parameters import load_params
import plots
import tboard
from SpecNormLayers_old import ConvSN2D, ConvSN2DTranspose, DenseSN


_SNlayertypes = [type(ConvSN2D(filters=2, kernel_size=2)), type(ConvSN2DTranspose(5, 5)), type(DenseSN(1))]



class JCGAN:
    
    def __init__(self, configtag, expDir):

        # Load hyperparmeters
        self.configtag = configtag
        logging.info('Parameters:')
        self.init_params(configtag)
        self.expDir = expDir

        # Import slices
        self.real_imgs = np.load('./data/'+self.dataname+'_train.npy')
        self.val_imgs = np.load('./data/'+self.valname+'_val.npy')
        self.n_imgs = self.real_imgs.shape[0]
        if self.datafmt == 'channels_last':
            self.real_imgs = np.moveaxis(self.real_imgs, 1, -1)
            self.val_imgs = np.moveaxis(self.val_imgs, 1, -1)
        
        # Build networks
        self.discrim, self.genrtor = self.load_networks()

        # Compile discriminator so it can be trained separately
        loss_fns = self._get_loss()
        def mean_prob(y_true, y_pred):
            # metric to measure mean probability of D predictions (0=fake, 1=real)
            return K.mean(K.sigmoid(y_pred))

        self.discrim.compile(loss=loss_fns['D'], optimizer=keras.optimizers.Adam(lr=self.D_lr, beta_1=0.5), metrics=[mean_prob])

        # Set up generator to use Jacobian clamping
        self.lam_max = 20.
        self.lam_min = 1.
        self.eps = 1.

        z = Input(shape=(1,self.noise_vect_len))
        eps = Input(shape=(1,self.noise_vect_len))
        G_z = self.genrtor(z)
        zp = keras.layers.add([z, eps])
        G_zp = self.genrtor(zp)
        logging.info('zp: '+str(zp.shape))
        dG = tf.norm(tf.squeeze(G_z - G_zp, axis=[-1]), axis=(1,2)) # NHWC format
        logging.info('dG: '+str(dG.shape))
        dz = tf.norm(tf.squeeze(eps, axis=[1]), axis=-1)
        logging.info('dz: '+str(dz.shape))
        Q = Lambda(lambda inputs: inputs[0] / inputs[1])([dG, dz])
        logging.info('Q: '+str(Q.shape))
        l_max = K.constant(self.lam_max, name='lam_max')
        l_min = K.constant(self.lam_min, name='lam_min')
        logging.info('lmin, lmax : '+str(l_min.shape) + ', '+str(l_max.shape))
        l_max_diff = keras.layers.maximum([Q - l_max, tf.zeros_like(Q)])
        l_min_diff = keras.layers.minimum([Q - l_min, tf.zeros_like(Q)])
        logging.info('lmin_diff, lmax_diff: '+str(l_min_diff.shape)+', '+str(l_max_diff.shape))
        Lmax = K.pow(l_max_diff, 2.)
        Lmin = K.pow(l_min_diff, 2.)
        L = keras.layers.Add()([Lmax, Lmin])
        logging.info('Lmax, Lmin, L: '+str(Lmax.shape)+', '+str(Lmin.shape)+', '+str(L.shape))
        
        self.discrim.trainable = False
        decision = self.discrim(G_z)
        self.stacked = Model(inputs=[z, eps], outputs=[decision])
        self.stacked.summary()
        self.stacked.compile(loss=loss_fns['G'](JC_loss=L), optimizer=keras.optimizers.Adam(lr=self.G_lr, beta_1=0.5))

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


    def _get_loss(self):
        
        Ldict = {}
        if self.loss == 'binary_crossentropy':
            def crossentropy_JC(JC_loss):
                def loss_func(y_true, y_pred):
                    loss =  K.mean(K.binary_crossentropy(y_true, y_pred, from_logits=True), axis=-1)
                    logging.info(loss.shape)
                    loss += JC_loss
                    logging.info(loss.shape)
                    return loss 
                return loss_func
            def my_crossentropy(y_true, y_pred):
                return K.mean(K.binary_crossentropy(y_true, y_pred, from_logits=True), axis=-1)

            Ldict['D'] = my_crossentropy
            Ldict['G'] = crossentropy_JC
        elif self.loss == 'hinge': # hinge loss is untested
            def Ghinge(ytrue, ypred):
                return -K.mean(ypred, axis=-1)
            Ldict['D'] = keras.losses.hinge
            Ldict['G'] = Ghinge
        return Ldict

    def build_discriminator(self):
        
        dmodel = Sequential()
        for lyrIdx in range(self.nlayers):
            if self.doubleconv:
                dmodel.add(self._conv_layer_type(1, lyrIdx))
            dmodel.add(self._conv_layer_type(self.convstride, lyrIdx)) 
            dmodel.add(BatchNormalization(epsilon=1e-5, momentum=0.9, axis=self.bn_axis))
            dmodel.add(LeakyReLU(alpha=self.alpha))
        dmodel.add(Flatten(data_format=self.datafmt))
        dmodel.add(self._dense_layer_type(1, shape_spec=False))
        if self.loss == 'hinge':
            act = 'tanh'
        else:
            act = 'sigmoid'
        #dmodel.add(Activation(act)) # no sigmoid activation if using logits directly 
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
                lyr = DenseSN(num, input_shape=shape_spec)
            else:
                lyr = Dense(num, input_shape=shape_spec)
        else:
            if self.specnorm:
                lyr = DenseSN(num)
            else:
                lyr = Dense(num)
        return lyr


    def _conv_layer_type(self, stride, lyrIdx):

        lyr = None
        if lyrIdx == 0:
            if self.specnorm:
                lyr = ConvSN2D(filters=self.nconvfilters[lyrIdx], kernel_size=self.convkern, strides=stride, 
                               padding='same', input_shape=self.imshape, data_format=self.datafmt)
            else:
                lyr = Conv2D(filters=self.nconvfilters[lyrIdx], kernel_size=self.convkern, strides=stride, 
                             padding='same', input_shape=self.imshape, data_format=self.datafmt)
        else:
            if self.specnorm:
                lyr = ConvSN2D(filters=self.nconvfilters[lyrIdx], kernel_size=self.convkern, strides=stride,
                               padding='same', data_format=self.datafmt)
            else:
                lyr = Conv2D(filters=self.nconvfilters[lyrIdx], kernel_size=self.convkern, strides=stride,
                             padding='same', data_format=self.datafmt)
        return lyr



    def _deconv_layer_type(self, stride, lyrIdx):

        lyr = None
        if self.specnorm:
            lyr = ConvSN2DTranspose(self.ndeconvfilters[lyrIdx], self.convkern, strides=stride,
                                    padding='same', data_format=self.datafmt)
        else:
            lyr = Conv2DTranspose(self.ndeconvfilters[lyrIdx], self.convkern, strides=stride,
                                  padding='same', data_format=self.datafmt)
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
        self.real = 1
        if self.loss == 'hinge': # hinge loss is untested
            self.fake = -1
        else:
            self.fake = 0

    def train_epoch(self, shuffler, num_batches, epochIdx):
        
        d_losses = []
        d_real_losses = []
        d_fake_losses = []
        g_losses = []

        for batch in range(num_batches):
            iternum = (epochIdx*num_batches + batch)
            t1 = time.time()

            # generate fakes
            real_img_batch = self.real_imgs[shuffler[batch*self.batchsize:(batch+1)*self.batchsize]]
            noise_vects = np.random.normal(loc=0.0, size=(self.batchsize, 1, self.noise_vect_len))
            eps = np.random.normal(loc=0.0, size=(self.batchsize, 1, self.noise_vect_len))*self.eps
            fake_img_batch = self.genrtor.predict(noise_vects)

            reals = self.real*np.ones((self.batchsize,1))
            fakes = self.fake*np.ones((self.batchsize,1))
            labelflip = self.label_flip
            for i in range(reals.shape[0]):
                if np.random.uniform(low=0., high=1.0) < labelflip:
                    reals[i,0] = self.fake
                    fakes[i,0] = self.real
            
            # train discriminator
            for iters in range(self.DG_update_ratio//2):
                discr_real_loss = self.discrim.train_on_batch(real_img_batch, reals)
                discr_fake_loss = self.discrim.train_on_batch(fake_img_batch, fakes)
                discr_loss = 0.5*(discr_real_loss[0]+discr_fake_loss[0])
                d_losses.append(discr_loss)
                d_real_losses.append(discr_real_loss[0])
                d_fake_losses.append(discr_fake_loss[0])

            # train generator via stacked model
            genrtr_loss = self.stacked.train_on_batch([noise_vects, eps], self.real*np.ones((self.batchsize,1)))
            t2 = time.time()

            g_losses.append(genrtr_loss)

            if batch%self.print_batch == 0:
                logging.info("| --- batch %d of %d --- |"%(batch + 1, num_batches))
                logging.info("|Discr real prob=%f, fake prob=%f"%(discr_real_loss[1], discr_fake_loss[1]))
                logging.info("|Discriminator: loss=%f"%(discr_loss)
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
                    # Monitor histrograms of weights for conv and dense layers
                    Dlayerdict = {layer.name:layer.get_weights()[0] for layer in self.discrim.layers[1].layers \
                                                                    if 'conv' in layer.name or 'dense' in layer.name}
                    Glayerdict = {layer.name:layer.get_weights()[0] for layer in self.genrtor.layers[1].layers \
                                                                    if 'conv' in layer.name or 'dense' in layer.name}

                    self.TB_Dwts.on_epoch_end(self, iternum, Dlayerdict)
                    self.TB_Gwts.on_epoch_end(self, iternum, Glayerdict)

                if self.grad_hists:
                    # Monitor histograms of gradients -- VERY slow and crashes for big networks
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
                    # Update best chi-square score and save checkpoint
                    self.bestchi = chisq
                    self.genrtor.save(self.expDir+'models/g_cosmo_best.h5')
                    self.discrim.save(self.expDir+'models/d_cosmo_best.h5')
                    logging.info("BEST saved at %d"%iternum)


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








