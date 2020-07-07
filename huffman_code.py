#%%
import os
os.getcwd()
# os.chdir(os.path.join(os.getcwd(), 'C:/works/PythonCode/DataCompression'))
import queue
import numpy as np
import matplotlib.pyplot as plt
import json
#
# Image processing
# PIL
from PIL import ImageFont, ImageDraw, Image


#%%
# Define how to count grey level frequency of image
def char_freq(ndarray):
    '''Count grey level frequency of image
    
    Returns
        freq: list ot tuple represents (level frequency, level value)
    '''
    freq = []
    
    ndarray = ndarray.flatten(order='K')  # shape:(height x width x channel)
    bins = np.bincount(ndarray)
    idx = np.nonzero(bins)[0]  # idx of non-zero elements
    
    for i in zip(bins[idx], idx):
        freq.append(i)
        
    return freq

#%%
# Define HuffmanNode containing its child nodes
class HuffmanNode(object):
    """HuffmanNode containing its child nodes, the intance may be a huge binary tree.
    """
    def __init__(self, left=None, right=None, root=None):
        self.left = left  # will be (freq: node) or (freq: char)
        self.right = right
    def children(self):
        return ((self.left, self.right))

#%%
def create_tree(frequencies):
    p = queue.PriorityQueue()
    for value in frequencies:    # 1. Create a leaf node for each symbol
        p.put(value)             #    and add it to the priority queue
    while p.qsize() > 1:         # 2. While there is more than one node
        l, r = p.get(), p.get()  # 2a. take out two lowest nodes to combine
        node = HuffmanNode(l, r) # 2b. create parent node with children (l,r)
        epsilon = np.random.uniform(high=1e-7)  # prevent same freq error
        p.put((l[0] + r[0] + epsilon, node)) # 2c. add parent (freq, node) to queue
    return p.get()               # 3. tree is complete - return root node

#%%
# Recursively walk the tree down to the leaves,
# assigning a code value to each symbol
def walk_tree(node, prefix="", code=None):
    """Pass a HuffmanNode or charactor to get the coding list.
    
    Arguments:
        node is a tuple(freq, HuffmanNode or character)
        
    Return:
        dict of each charactor and its code {char_i: code_i}
    """
    code = {}
    if isinstance(node[1], HuffmanNode):  # if input is not the lowest node(str)
        code1 = walk_tree(node[1].left, '0', code.copy())  # extract left child
        code2 = walk_tree(node[1].right, '1', code.copy())
        if len(code1) > 0:
            for k, v in code1.items():
                code[k] = prefix + v  # add prefix(0 or 1) to the code
        if len(code2) > 0:
            for k, v in code2.items():
                code[k] = prefix + v
    else:  # input is a charactor(or single digit)
        code[node[1]] = prefix  # give the charactor 0 or 1 as code
    return code

# prefix是非最底層的子節點與父節點箭頭上賦予的數字


# %%
# encoder

def img_encoder(img):
    freq = char_freq(img)  # freq
    node = create_tree(freq)  # tree
    code = walk_tree(node)  # encode list
    img = img.copy().flatten(order='K')
    img2 = [str(i) for i in img.tolist()]

    for k, v in code.items():
        mask = img == k
        idx = np.where(mask)[0]
        for i in idx:
            img2[i] = v

    return ''.join(img2), node, code, freq


# a, b, c, d = img_encoder(img)
# %%
# Find decoded data
def traverse(encoded, node, i=0):
    if encoded[i] == "0":
        if isinstance(node[1].left[1], HuffmanNode):
            return traverse(encoded, node[1].left, i+1)
        else:
            # print('left', node[1].left[1])
            return node[1].left[1]
    else:
        if isinstance(node[1].right[1], HuffmanNode):
            return traverse(encoded, node[1].right, i+1)
        else:
            # print('right', node[1].right[1])
            return node[1].right[1]


#%%
def img_decoder(encoded, node, code, shape):
    decoded = []
    while len(encoded) > 0:
        value = traverse(encoded, node)
        decoded.append(value)
        encoded = encoded[len(code[value]):]
    return np.array(decoded).reshape(shape)

#%%
# Save as txt
# im_save = im.astype('uint8')
# im_save = ''.join([str(i) for i in im.tolist()])
# # np.savetxt('./data/im.txt', im_save)

# with open('./data/im_encode.txt', 'w') as f:
#     f.write(encoded)
#%%
# Import image and sort
# im = Image.open('./data/00003.png') #.convert('RGB')
# im = np.asarray(im)
# im.dtype
# plt.imshow(im)#, cmap='gray')
# encoded, node, code, freq = img_encoder(im)
# # with open('./data/en.txt', 'w') as f:
# #     f.write(encoded)
    
# with open('./data/en.txt', 'r') as f:
#     encoded1 = f.readlines()[0]
# encoded1 == encoded
# decode = img_decoder(encoded1, node, code)
# decode.astype('uint8')
# plt.imshow(decode)
# freq = char_freq(im)  # freq
# node = create_tree(freq)  # tree
# code = walk_tree(node)  # encode list
# # print pixel levels, freq and code
# for i in sorted(freq, reverse=True):
#     print(i[1], '{:6.2f}'.format(i[0]), code[i[1]])
# encoded = encoder(im, code)
# decoded = decoder(encoded)
# im_decoded = np.array(decoded).reshape(250, 200, 3)
# plt.imshow(im_decoded)

#%%
# Compression rate

# len(encoded)  / (250*200*3*8)

#
# 512*483*3*8
# 5935104
# a = np.random.randint(0,9,(5,5,3))
# a.shape

# 28*28*8
