import numpy as np
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.append('./utils')
sys.path.append('./networks')
import logging
import logging_utils
import JCGANbuild
import parameters


configtag = sys.argv[1]
run_num = sys.argv[2]

baseDir = './expts/'+configtag+'/'
expDir = baseDir+'run'+str(run_num)+'/'
if not os.path.isdir(baseDir):
    os.mkdir(baseDir)
if not os.path.isdir(expDir):
    os.mkdir(expDir)
    os.mkdir(expDir+'models')
else:
    print("Experiment directory %s already exists, exiting"%expDir)
    sys.exit()


#Set up logger
logging_utils.config_logger(log_level=logging.INFO)
logging_utils.log_to_file(logger_name=None, log_filename=expDir+'train.log')


# Build GAN
GAN = JCGANbuild.JCGAN(configtag, expDir)


Nepochs = GAN.Nepochs
Nbatches = GAN.n_imgs // GAN.batchsize

for epoch in np.arange(GAN.start, Nepochs+GAN.start):
    logging.info("| ******************************* Epoch %d of %d ******************************* |"%(epoch+1, Nepochs+GAN.start))
    shuff_idxs = np.random.permutation(GAN.n_imgs)
    GAN.train_epoch(shuff_idxs, Nbatches, epoch)
    
    if (epoch+1)%5==0: 
        GAN.genrtor.save(GAN.expDir+'models/g_cosmo%04d.h5'%(epoch))
        GAN.discrim.save(GAN.expDir+'models/d_cosmo%04d.h5'%(epoch))

GAN.genrtor.save(GAN.expDir+'models/g_cosmo_last.h5')
GAN.discrim.save(GAN.expDir+'models/d_cosmo_last.h5')

logging.info('DONE')



