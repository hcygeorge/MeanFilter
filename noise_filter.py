#%%
# Data science tools
import numpy as np

#%%
# Mean filter
def mean_filter(image, filter_size=3, method='A'):
    '''One channel images mean filter'''
    out = np.copy(image)  # new copy of data
    # if len(image.shape) == 2:  # if image is 2D array
    #     image_3d = np.expand_dims(image, axis=-1)  # add new dimension as channel
    # for channel in range(1):  # image_3d.shape[-1]
    #     image = image_3d[:, :, channel]
    for row in range(image.shape[0] - filter_size + 1):
        for col in range(image.shape[1] - filter_size + 1):
            block = image[row:row + filter_size, col:col + filter_size]  # the region being filtered
            if method == 'A':
                f_hat = np.sum(block) / filter_size**2  # arithmetic mean
            elif method == 'G':
                f_hat = np.exp(np.sum(np.log(block + 1e-10)) / filter_size**2)  # geometric mean
                # f_hat = np.prod(block)**(1 / filter_size**2)
            elif method == 'H':
                f_hat = filter_size**2 / np.sum(1/(block + 1e-10))  # harmonic mean

            out[row + filter_size//2, col + filter_size//2] = f_hat
    return out

# testing
# np.random.seed(7)
# gar01 = np.random.randint(1, 3, (4, 4))
# print(gar01)
# len(gar01.shape)
# gar01.shape
# gar01[:, :, 0].shape
# gar02 = np.array((gar01, gar01, gar01))
# gar02.shape

#%%
# Median filter

def median_filter(image, filter_size=3):
    out = np.copy(image)  # create a new copy of image
    for row in range(out.shape[0] - filter_size + 1):
        for col in range(out.shape[1] - filter_size + 1):
            block = out[row:row + filter_size, col:col + filter_size]
            f_hat = np.median(block.reshape(1, -1))  # median
            out[row + filter_size//2, col + filter_size//2] = f_hat
    return out

#%%
# Contraharmonic
def filter_contraharmonic(image, filter_size=3, Q=1):
    out = np.copy(image).astype('int64')  # create a new copy of image
    count = 0
    Q = 1
    for row in range(out.shape[0] - filter_size + 1):
        for col in range(out.shape[1] - filter_size + 1):
            block = out[row:row + filter_size, col:col + filter_size]
            f_hat = np.sum(block**(Q + 1)) / (np.sum(block**Q) + 1e-10)
            out[row + filter_size//2, col + filter_size//2] = f_hat
        count += 1    
    return out
#%%

#%%
# RGB mean filter
# img = Image.open('./data/ntpu_logo.png').convert('RGB')
# img = np.array(img)
# plt.imshow(img)
# np.expand_dims(img, axis=-1).shape

#%%