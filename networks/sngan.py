import numpy as np
import keras
from keras.layers import *
from keras.activations import relu
from keras.models import Model, Sequential
from keras.models import load_model
import tensorflow as tf
import time
import sys
sys.path.append('./utils')
sys.path.append('./networks')
import logging
import logging_utils
from parameters import load_params
import dcgan
import plots
import tboard
from SpecNormLayers import ConvSN2D, ConvSN2DTranspose, DenseSN

class SNGAN:
    
    def __init__(self, configtag):

        # Load hyperparmeters
        self.configtag = configtag
        self.init_params(configtag)

        # Import slices
        self.real_imgs = np.load('./data/'+self.dataname+'_train.npy')
        self.val_imgs = np.load('./data/'+self.dataname+'_val.npy')
        self.n_imgs = self.real_imgs.shape[0]

        # Build networks
        self.discrim = self.build_discriminator()
        self.genrtor = self.build_generator()

	# Compile discriminator so it can be trained separately
        self.discrim.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])

        # Stack generator and discriminator networks together and compile
        z = Input(shape=(1,self.noise_vect_len))
        genimg = self.genrtor(z)
        self.discrim.trainable = False
        decision = self.discrim(genimg)
        self.stacked = Model(z, decision)
        self.stacked.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5))

        # Setup tensorboard stuff
        self.TB_genimg = tboard.TboardImg('genimg')
        self.TB_pixhist = tboard.TboardImg('pixhist')
        self.TB_scalars = tboard.TboardScalars()


    def build_discriminator(self):
        dmodel = Sequential()
        for lyrIdx in range(self.nlayers): 
            if lyrIdx==0:
                 # feed input dims 
                 dmodel.add(ConvSN2D(filters=self.nconvfilters[lyrIdx], kernel_size=self.convkern, strides=self.convstride, 
                                   padding='same', input_shape=(1, self.img_dim, self.img_dim), data_format=self.datafmt))
            else:
                 dmodel.add(ConvSN2D(filters=self.nconvfilters[lyrIdx], kernel_size=self.convkern, strides=self.convstride,
                                   padding='same', data_format=self.datafmt)) 
            dmodel.add(BatchNormalization())
            dmodel.add(LeakyReLU(alpha=self.alpha))
        dmodel.add(Flatten())
        dmodel.add(Dense(1))
        dmodel.add(Activation('sigmoid'))
        dmodel.summary()
        img = Input(shape=(1,self.img_dim,self.img_dim))
        return Model(img, dmodel(img))


    def build_generator(self):
        gmodel = Sequential()
        gmodel.add(DenseSN(8*8*(2*self.ndeconvfilters[0]), input_shape=(1,self.noise_vect_len)))
        gmodel.add(BatchNormalization())
        gmodel.add(ReLU())
        gmodel.add(Reshape((2*self.ndeconvfilters[0], 8, 8)))
        for lyrIdx in range(self.nlayers):
            gmodel.add(ConvSN2DTranspose(self.ndeconvfilters[lyrIdx], self.convkern, strides=self.convstride, 
                                       padding='same', data_format=self.datafmt))
            if lyrIdx == self.nlayers - 1:
                gmodel.add(Activation('tanh')) # last layer is deconv+tanh
            else:
                gmodel.add(BatchNormalization()) # hidden layers are deconv+batchnorm+relu
                gmodel.add(ReLU())
        gmodel.summary()
        noise = Input(shape=(1,self.noise_vect_len))        
        return Model(noise, gmodel(noise))


    def init_params(self,configtag):
        params = load_params('./config.yaml', configtag)
        self.dataname = params['dataname']
        self.expDir = params['expDir']
        self.img_dim = params['img_dim']
        self.noise_vect_len = params['noise_vect_len']
        self.nlayers = params['nlayers']
        self.convkern = params['convkern']
        self.convstride = params['convstride']
        self.nconvfilters = params['nconvfilters']
        self.ndeconvfilters = params['ndeconvfilters']
        self.label_flip = params['label_flip']
        self.batchsize = params['batchsize']
        self.print_batch = params['print_batch']
        self.print_epoch = params['print_epoch']
        self.histwindow = params['histwindow']
        self.cscale = params['cscale']
        self.alpha = 0.2
        self.start = 0
        self.datafmt = 'channels_first'

    def train_epoch(self, shuffler, num_batches, epochIdx):
        d_losses = []
        d_real_losses = []
        d_fake_losses = []
        g_losses = []

        for batch in range(num_batches):
            t1 = time.time()

            # generate fakes
            real_img_batch = self.real_imgs[shuffler[batch*self.batchsize:(batch+1)*self.batchsize]]
            noise_vects = np.random.normal(loc=0.0, size=(self.batchsize, 1, self.noise_vect_len))
            fake_img_batch = self.genrtor.predict(noise_vects)

            reals = np.ones((self.batchsize,1))
            fakes = np.zeros((self.batchsize,1))
            for i in range(reals.shape[0]):
                if np.random.uniform(low=0., high=1.0) < self.label_flip:
                    reals[i,0] = 0
                    fakes[i,0] = 1

            # train discriminator
            discr_real_loss = self.discrim.train_on_batch(real_img_batch, reals)
            discr_fake_loss = self.discrim.train_on_batch(fake_img_batch, fakes)
            discr_loss = [0.5*(discr_real_loss[0]+discr_fake_loss[0]),
                          0.5*(discr_real_loss[1]+discr_fake_loss[1])]
            # train generator via stacked model
            genrtr_loss = self.stacked.train_on_batch(noise_vects, np.ones((self.batchsize,1)))
            t2 = time.time()

            d_losses.append(discr_loss[0])
            d_real_losses.append(discr_real_loss[0])
            d_fake_losses.append(discr_fake_loss[0])
            g_losses.append(genrtr_loss)

            # Tensorboard stuff
            iternum = batch + self.batchsize*epochIdx
            
            t2 = time.time()

            if batch%self.print_batch == 0:
                logging.info("| --- batch %d of %d --- |"%(batch + 1, num_batches))
                logging.info("|Discr real acc=%f, fake acc=%f"%(discr_real_loss[1], discr_fake_loss[1]))
                logging.info("|Discriminator: loss=%f, accuracy = %f"%(discr_loss[0], discr_loss[1]))
                logging.info("|Generator: loss=%f"%(genrtr_loss))
                logging.info("|Time: %f"%(t2-t1))

        self.TB_genimg.on_epoch_end(epochIdx, self, logs={})
        chisq = self.TB_pixhist.on_epoch_end(epochIdx, self, logs={})
        scalars = {'d_loss':np.mean(d_losses), 'd_real_loss':np.mean(d_real_losses),
                   'd_fake_loss':np.mean(d_fake_losses), 'g_loss':np.mean(g_losses), 'chisq':chisq}
        self.TB_scalars.on_epoch_end(self, epochIdx, scalars)

        return np.mean(d_losses), np.mean(d_real_losses), np.mean(d_fake_losses), np.mean(g_losses)















