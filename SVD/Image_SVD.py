# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 14:29:30 2017

@author: saTan-Y
"""

from PIL import Image
import numpy as np
import time

def rebuild_image(u, sigma, v, p):
    m, n = len(u), len(v)
    image = np.zeros((m,n))
    sigmasum = int(sum(sigma))
    sigmacount = 0
    k = 0
    
    while sigmacount <= sigmasum*p:
        uk = u[:,k].reshape(m,1)
        vk = v[k].reshape(1,n)
        image += sigma[k]*np.dot(uk,vk)
        sigmacount += sigma[k]
        k += 1
    print('number of sigma: ', k)
    image[image<0] = 0
    image[image>255] = 255
    return np.rint(image).astype('uint8')
    
if __name__ == '__main__':
    t0 = time.time()
    img = Image.open('james.jpg', 'r')
    a = np.array(img)
    
    ur, sigmar, vr = np.linalg.svd(a[:, :, 0])
    ug, sigmag, vg = np.linalg.svd(a[:, :, 1])
    ub, sigmab, vb = np.linalg.svd(a[:, :, 2])
    
    for p in np.arange(0.1, 1, 0.1):
        
        R = rebuild_image(ur, sigmar, vr, p)
        
        G = rebuild_image(ug, sigmag, vg, p)
        
        B = rebuild_image(ub, sigmab, vb, p)
        
        temp = np.stack((R, G, B), 2)
        Image.fromarray(temp).save('svd_' + str(p*100) + '.jpg')
    t1 = time.time()
    print('Elapsed time is ', t1 - t0)
    