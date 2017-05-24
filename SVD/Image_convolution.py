# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 16:43:02 2017

@author: saTan-Y
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
import os

def image_convolution(image, weight):
    m, n = image.shape
    h, w = weight.shape
    temp = np.concatenate((np.zeros((m, 1)), image, np.zeros((m, 1))), axis=1)
    image_extension = np.concatenate((np.zeros((1, n+2)), temp, np.zeros((1, n+2))))
    new_image = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            new_image[i, j] = np.sum(image_extension[i:i+h, j:j+w] * weight)
    new_image = new_image.clip(0, 255)
    return np.rint(new_image).astype('uint8')
    
if __name__ == '__main__':
    t0 = time.time()
    image_file = Image.open('.\\7.Package\\7.lena.png')
    a = np.array(image_file)
    outpath = '.\\conPic\\'
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    
    soble_x = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))
    soble_y = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]))
    soble = np.array(([-1, -1, 0], [-1, 0, 1], [0, 1, 1]))
    prewitt_x = np.array(([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]))
    prewitt_y = np.array(([-1, -1,-1], [0, 0, 0], [1, 1, 1]))
    prewitt = np.array(([-2, -1, 0], [-1, 0, 1], [0, 1, 2]))
    laplacian = np.array(([0, -1, 0], [-1, 4, -1], [0, -1, 0]))
    laplacian2 = np.array(([-1, -1, -1], [-1, 8, -1], [-1, -1, -1]))
    
    weight_list = ('soble_x', 'soble_y', 'soble', 'prewitt_x', 'prewitt_y', 'prewitt', 'laplacian', 'laplacian2')
    for name in weight_list:
#        print('Start: ')
        print(name, 'R', end='')
        R = image_convolution(a[:, :, 0], eval(name))
        print('G', end='')
        G = image_convolution(a[:, :, 1], eval(name))
        print('B', end='')
        B = image_convolution(a[:, :, 2], eval(name))
        I = 255 - np.stack((R, G, B), 2)
        Image.fromarray(I).save(outpath + name + '.png')
    t1 = time.time()
    print('Elapsed time is: ', t1-t0)
        
    