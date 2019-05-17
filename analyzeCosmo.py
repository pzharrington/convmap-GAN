import numpy as np
import time
import keras
from keras.layers import *
from keras.activations import relu
from keras.models import Model, Sequential
from keras.models import load_model
import keras.backend as K
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LogNorm
import sys
import tensorflow as tf

sys.path.append('./utils/')
sys.path.append('./networks/')
import plots
from SpecNormLayers import DenseSN, ConvSN2D, ConvSN2DTranspose



def my_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred, from_logits=True), axis=-1)

# custom_layers = None
custom_layers = {'DenseSN':DenseSN, 'ConvSN2D':ConvSN2D, 'ConvSN2DTranspose':ConvSN2DTranspose, 
                 'my_crossentropy':my_crossentropy}

# Import slices
real_imgs = np.load('./data/fullcrop_val.npy')
n_imgs = real_imgs.shape[0]
noise_vect_len = 64
print(real_imgs.shape)

# Uncomment to set BatchNorm statistics to current batch
#K.set_learning_phase(1)

# Load weights
genrtor = load_model('./expts/bigSN-run7/models/g_cosmo_best.h5', custom_objects=custom_layers)
discrim = load_model('./expts/bigSN-run7/models/d_cosmo_best.h5', custom_objects=custom_layers)
discrim.summary()
genrtor.summary()
lossfn = 'binary_crossentropy'

discrim.compile(loss=lossfn, optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])

z = Input(shape=(1,noise_vect_len))
genimg = genrtor(z)
discrim.trainable = False
decision = discrim(genimg)
stacked = Model(z, decision)
stacked.compile(loss=lossfn, optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5))



noise_vects1 = np.random.normal(loc=0.0, size=(10, 1, noise_vect_len))


reals = real_imgs[:10,:,:,:]
fakes = genrtor.predict(noise_vects1)

#plots.save_realimg_grid(real_imgs, Xterm=True, scale='lin')
#plots.save_img_grid(genrtor, noise_vect_len, 0, Xterm=True, scale='pwr')
plots.save_img_grid(genrtor, noise_vect_len, 0, Xterm=True, scale='pwr')
plots.save_img_grid(genrtor, noise_vect_len, 0, Xterm=True, scale='pwr')
#wdw = [-1.1, 1.1, 1e-4, 3e4]
plots.pix_intensity_hist(real_imgs, genrtor, noise_vect_len, 'lin', Xterm=True)
plt.show()

'''
print(discrim.predict(reals))
print(discrim.predict(fakes))
print(stacked.predict(noise_vects1))

batchsize = 64
real_batch = real_imgs[27:27+batchsize,:,:,:]
noise_vects = np.random.normal(loc=0.0, size=(batchsize, 1, noise_vect_len))
fake_batch = genrtor.predict(noise_vects)
reallabs = np.ones((batchsize, 1))
fakelabs = np.zeros((batchsize, 1))


discr_real_loss = discrim.test_on_batch(real_batch, reallabs)
discr_fake_loss = discrim.test_on_batch(fake_batch, fakelabs)
g_loss = stacked.test_on_batch(noise_vects, reallabs)

print(discr_real_loss)
print(discr_fake_loss)
print(g_loss)


discr_real_loss = discrim.train_on_batch(real_batch, reallabs)
discr_fake_loss = discrim.train_on_batch(fake_batch, fakelabs)
g_loss = stacked.train_on_batch(noise_vects, reallabs)

print(discr_real_loss)
print(discr_fake_loss)
print(g_loss)

'''




