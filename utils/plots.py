import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm

def invtransf_a(imgs, a, scaling='dyn'):
    if scaling == 'lin':
        return a*imgs
    else:
        return np.divide(a*imgs, 1. - imgs)

def pix_intensity_hist(vals, generator, noise_vector_length, scaling, fname=None, Xterm=True, window=None):
    num = len(vals)
    samples = generator.predict(np.random.normal(size=(num,1,noise_vector_length)))
    samples = invtransf_a(samples, 1., scaling='lin')
    valhist, bin_edges = np.histogram(vals.flatten(), bins=25)
    samphist, _ = np.histogram(samples.flatten(), bins=bin_edges)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    plt.figure()
    plt.errorbar(centers, valhist, yerr=np.sqrt(valhist), fmt='o-', label='validation')
    plt.errorbar(centers, samphist, yerr=np.sqrt(samphist), fmt='o-', label='generated')
    plt.gca().set_yscale('log')
    plt.legend(loc='upper right')
    if window:
        plt.axis(window)
    if Xterm:
        plt.draw()
    else:
        plt.savefig(fname, format='png')
        plt.close()

    return np.sum(np.divide(np.power(valhist - samphist, 2.0), valhist))



def save_img_grid(generator, noise_vector_length, fname=None, Xterm=True, scale='lin'):
    imgs_per_side = 2
    samples = generator.predict(np.random.normal(size=(imgs_per_side**2,1,noise_vector_length)))
    genimg_sidelen = samples.shape[2]

    tot_len=imgs_per_side*genimg_sidelen
    gridimg = np.zeros((tot_len, tot_len))
    cnt = 0
    for i in range(imgs_per_side):
        for j in range(imgs_per_side):
            gridimg[i*genimg_sidelen:(i+1)*genimg_sidelen, j*genimg_sidelen:(j+1)*genimg_sidelen] \
                = samples[cnt,0,:,:]
            cnt += 1
    plt.figure(figsize=(5,4))
    if scale == 'pwr':
        imgmap = plt.pcolormesh(gridimg, norm=LogNorm(vmin=1e-4, vmax=0.5),
                                cmap='Blues') # Log normalized color scale
    else:
        imgmap = plt.imshow(gridimg, cmap='Blues') # Linear color scale
    plt.colorbar(imgmap)
    plt.plot([tot_len//2, tot_len //2], [0, tot_len], 'k-', linewidth='0.6')
    plt.plot([0, tot_len], [tot_len//2, tot_len //2], 'k-', linewidth='0.6')
    plt.axis([0, tot_len, 0 , tot_len])
    if Xterm:
        plt.draw()
    else:
        plt.savefig(fname, format='png')
        plt.close()

def save_realimg_grid(reals, Xterm=True, scale="lin"):
    imgs_per_side = 2
    samples = reals[np.random.randint(reals.shape[0], size=(4))]
    smpimg_sidelen = samples.shape[2]

    tot_len=imgs_per_side*smpimg_sidelen
    gridimg = np.zeros((tot_len, tot_len))
    cnt = 0
    for i in range(imgs_per_side):
        for j in range(imgs_per_side):
            gridimg[i*smpimg_sidelen:(i+1)*smpimg_sidelen, j*smpimg_sidelen:(j+1)*smpimg_sidelen] \
                = samples[cnt,0,:,:]
            cnt += 1
    plt.figure(figsize=(5,4))
    if scale == 'pwr':
        imgmap = plt.pcolormesh(gridimg, norm=LogNorm(vmin=1e-4, vmax=0.5),
                                cmap='Blues')  # Power-law normalized color scale
    else:
        imgmap = plt.imshow(gridimg, cmap='Blues') # Linear color scale
    plt.colorbar(imgmap)
    plt.plot([tot_len//2, tot_len //2], [0, tot_len], 'k-', linewidth='0.6')
    plt.plot([0, tot_len], [tot_len//2, tot_len //2], 'k-', linewidth='0.6')
    plt.axis([0, tot_len, 0 , tot_len])
    plt.title('real imgs')
    if Xterm:
        plt.draw()
    else:
        plt.savefig('./imgs/real_example.png')
        plt.close()

