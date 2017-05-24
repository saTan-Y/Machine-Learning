# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 14:29:12 2017

@author: saTan-Y
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import time
#from scipy import stats

t0 = time.time()
def cal1(data):
    u1, u2, u3= 0.0, 0.0, 0.0
    n = len(data)
    for i in data:
        u1 += i
        u2 += i**2
        u3 += i**3
    u1 /= n
    u2 /= n
    u3 /= n
    sigma = math.sqrt(u2 - u1**2)
    return u1, sigma, u3
    
def cal2(data):
    u1, sigma, u3 = cal1(data)
    n = len(data)
    u4 = 0.0
    for j in data:
        u4 += (j - u1)**4
    u4 /= n
    skew = (u3 - 3*u1*sigma**2 - u1**3)
    kurt = u4 / sigma**4
    return u1, sigma, skew, kurt
    
if __name__ == '__main__':
    data = list(np.random.randn(10000))
    data2 = list(2 * np.random.randn(10000))
    data3 = [x for x in data if x>-0.5]
    data4 = list(np.random.uniform(0, 4, 10000))
    [u1, sigma1, skew1, kurt1] = cal2(data)
    [u2, sigma2, skew2, kurt2] = cal2(data2)
    [u3, sigma3, skew3, kurt3] = cal2(data3)
    [u4, sigma4, skew4, kurt4] = cal2(data4)       
    print('Four basical parameters are: u1 = %f, sigma1 = %f, skew1 = %f, kurt1 = %f' % (u1, sigma1,skew1, kurt1))
    print('Four basical parameters are: u2 = %f, sigma2 = %f, skew2 = %f, kurt2 = %f' % (u2, sigma2,skew2, kurt2))
    print('Four basical parameters are: u3 = %f, sigma3 = %f, skew3 = %f, kurt3 = %f' % (u3, sigma3,skew3, kurt3))
    print('Four basical parameters are: u4 = %f, sigma4 = %f, skew4 = %f, kurt4 = %f' % (u4, sigma4,skew4, kurt4))
    info1 = r'u1 = %f, sigma1 = %f, skew1 = %f, kurt1 = %f' % (u1, sigma1,skew1, kurt1)
    info2 = r'u2 = %f, sigma2 = %f, skew2 = %f, kurt2 = %f' % (u2, sigma2,skew2, kurt2)
    plt.hist(data, 50, normed = True, facecolor = 'r', alpha = 0.9)
    plt.hist(data4, 50, normed = True, facecolor = 'g', alpha = 0.7)
    plt.text(1, 0.42, info1, bbox = dict(facecolor = 'red', alpha = 0.7))
    plt.text(1, 0.4, info2, bbox = dict(facecolor = 'green', alpha = 0.7))    
    plt.grid(True)
    plt.show()
    
    