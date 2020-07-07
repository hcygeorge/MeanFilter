#%%
# Working directory
import os
os.chdir(os.path.join(os.getcwd(), 'C:/works/PythonCode/MeanFilter'))
os.getcwd()
#%%
# Built in tools
import pickle
import itertools
import math
from glob import glob  # find file path
from time import time  # time.time()
from collections import Counter, OrderedDict
#%%
# Image processing
from PIL import ImageFont, ImageDraw, Image
import cv2  # 2 means it uses C++ api, otherwise using C api 
print(cv2.__version__)  # 4.1.1
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
# Load single image
img = Image.open('./data/cat.jpg')
img = img.convert('L')
img = np.array(img)
plt.imshow(img)
plt.show()

#%%
# Load all the images in a directory
# Assign the directory of images 
img_dir = "./data"
# Extract the list of images' path
img_path = glob(img_dir + '/*')
# Extract the filenames of image and load images as array
img_name = []
img_array = []
for path in img_path:
    # get the filename and drop the file extension 
    img_name.append(path.split(sep = '\\')[-1][:-4])
    img = np.asarray(Image.open(path).convert('RGB'))
    img_array.append(img)

train_x = np.array(img_array)

# View images
for i in range(2):
    img = train_x[i]
    plt.title(str(img_name[i]))
    plt.imshow(img, cmap = plt.cm.gray)
    plt.show()


