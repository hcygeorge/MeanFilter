#%%
import numpy as np
img = np.array([[2, 4, 2, 2, 4],
                [0, 6, 7, 0, 8],
                [0, 4, 3, 6, 3],
                [5, 8, 7, 9, 7],
                [4, 4, 4, 8, 1]])

conv01 = np.array([[-1, 0, -2],
                    [-1, 2, 3],
                    [1, -1, 1]])

# conv02 = np.array([[0, -1, 0],
#                     [-1, 4, -1],
#                     [0, -1, 0]])

#%%
# Conv01
x = img
conv_filter = conv01
filter_size = 3
output_size = 5 - filter_size + 1
out = np.zeros([output_size, output_size])
for row in range(x.shape[0] - filter_size + 1):
    for col in range(x.shape[1] - filter_size + 1):
        block = x[row:row + filter_size, col:col + filter_size]
        conv = np.sum(np.multiply(block, conv_filter))
        out[row, col] = conv

out01 = out       

#%%
# ReLU01
x = out01
out02 = x*(x > 0)

#%%
# Conv02
x = out02
conv_filter = conv02
filter_size = 3
output_size = 6 - filter_size + 1
out = np.zeros([output_size, output_size])
for row in range(x.shape[0] - filter_size + 1):
    for col in range(x.shape[1] - filter_size + 1):
        block = x[row:row + filter_size, col:col + filter_size]
        conv = np.sum(np.multiply(block, conv_filter))
        out[row, col] = conv

out03 = out       

#%%
# ReLU02
x = out03
out04 = x*(x > 0)
out04
#%%
# Pooling
x = out04
input_size = 4
filter_size = 2
stride = 2
output_size = ((input_size - filter_size) // stride) + 1
out = np.zeros([output_size, output_size])
for row in range(0, x.shape[0] - filter_size + 1, stride):
    for col in range(0, x.shape[1] - filter_size + 1, stride):
        block = x[row:row + filter_size, col:col + filter_size]
        pool = np.max(block)
        out[row//2, col//2] = pool
        
out05 = out
out05
#%%
# FC01
x = out05
x = x.reshape(1, -1)
W01 = np.array([[0.5, -0.5, 0.5, -0.5],
                [0.5, -1.0, 1.0, -0.5]]).T
out = np.dot(x, W01)
out06 = out
out06
#%%
# Sigmoid
x = out06
out = 1 / (1 + np.exp(-x))
out07 = out
#%%
# FC2
x = out07
x = x.reshape(1, -1)
W02 = np.array([[-0.5, -1],
                [0.5, 1]]).T
out = np.dot(x, W02)
out08 = out
#%%
# Sigmoid
x = out08
out = 1 / (1 + np.exp(-x))
out09 = out
#%%
# Softmax
x = out09
np.exp(x[:, 0]) / np.sum(np.exp(x))
np.exp(x[:, 1]) / np.sum(np.exp(x))

# %%
(1 - 0.6537)*(0.6537 - 0.6537**2)*0.8176(1 - 0.8176)