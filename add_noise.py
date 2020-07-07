#%%
# Data science tools
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection  # drawing a collection of objects on the plot
from matplotlib.patches import Rectangle  # draw bbox
#%%
# Add Gaussian noise on images
def add_gaussian_noise(image, mean=0, std=0.1,
                       depth='8bits', show_img=False):
    out = np.copy(image)

    num_level = {'1bit': 2, '4bits': 2**4,
                '8bits': 2**8, '16bits': 2**16,
                'normalized': 2}
    
    out = out / (num_level['8bits'] - 1)  # normalize data to fix the effect of noise
    noise_gauss = np.random.normal(mean, std, size=out.shape)
    out += noise_gauss
    out = np.round(out*(num_level['8bits'] - 1), 0)
    if (out < 0).any() or (out > 1).any():  # limit the range of pixels in 0~1
        out = np.clip(out, 0, (num_level['8bits'] - 1))

    if show_img:
        plt.imshow(out, cmap='gray')

    return out.astype('uint8')
#%%
# Add Pepper noise on images
def add_impulse_noise(image, ratio_noise=0.05, ratio_salt=0.5,
                      depth='8bits', show_img=False):
    out = np.copy(image)
    
    num_level = {'1bit': 2, '4bits': 2**4,
                 '8bits': 2**8, '16bits': 2**16,
                 'normalized': 2}
    
    num_salt = np.ceil(out.size*ratio_noise*ratio_salt)
    coord_salt = [np.random.randint(0, i, int(num_salt)) for i in out.shape]  # [list of x-coord,list of y_coord]
    out[tuple(coord_salt)] = num_level[depth] - 1

    num_pepper = np.ceil(out.size*ratio_noise*(1 - ratio_salt))
    coord_pepper = [np.random.randint(0, i, int(num_pepper)) for i in out.shape]  # [list of x-coord,list of y_coord]
    out[tuple(coord_pepper)] = 0

    if show_img:
        plt.imshow(out, cmap='gray')
    
    return out
