# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 10:55:12 2017

@author: saTan-Y
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib as mpl

if __name__ == '__main__':
    t0 = time.time()
    stock_max, stock_min, stock_close, stock_amount = np.loadtxt('7.SH600000.txt', delimiter='\t', skiprows=2, usecols=(2, 3, 4, 5), unpack=True)
    N = 100
    print(N)
    
    n = 5
    weight1 = np.ones(n)
    weight1 /= weight1.sum()
    print(weight1)
    stock_1 = np.convolve(stock_close[:N], weight1, mode='valid')
    
    weight2 = np.exp(np.linspace(1, 0, 5))
    weight2 /= weight2.sum()
    print(weight2)
    stock_2 = np.convolve(stock_close[:N], weight2, mode='valid')
    
    t = np.arange(n-1, N)
    coe = np.polyfit(t, stock_2, 9)
    print(coe)
    estimation = np.polyval(coe, t)
    
    plt.figure()
    plt.plot(range(N), stock_close[:N], 'ro-', lw=2, label='original close price')
    plt.plot(t, stock_1, 'g-', lw=2, label='simple moving average')
    plt.plot(t, stock_2, 'b-', lw=2, label='exp moving average')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize = (9,6))
    plt.plot(range(N), stock_close[:N], 'ro-', lw=2, label='original close price')
    plt.plot(t, stock_2, 'g-', lw=2, label='exp moving average')
    plt.plot(t, estimation, 'b-', lw=2, label='exp moving average estimation')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
    
    t1 = time.time()
    print('Elapsed time is: ', t1-t0)
    