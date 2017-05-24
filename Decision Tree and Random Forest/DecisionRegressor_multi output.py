# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 15:24:56 2017

@author: saTan-Y
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

if __name__ == '__main__':
    t0 = time.time()
    N = 500
    x = np.random.rand(N)*8 - 4
    x.sort()
    y1 = np.sin(x) + 3 + np.random.randn(N) * 0.1
    y2 = np.cos(0.3*x) + np.random.randn(N) * 0.01
    y = np.vstack((y1, y2)).T
    x = x[:, None]
                 
    model = DecisionTreeRegressor(criterion='mse', max_depth=9)
    model.fit(x, y)
    x_test = np.linspace(-4, 4, 500)
    x_test = x_test[:, None]
    y_pred = model.predict(x_test)
    plt.scatter(y[:, 0], y[:, 1], c='r', s=40, label='train')
    plt.scatter(y_pred[:, 0], y_pred[:, 1], c='g', s=40, label='test')
    plt.legend(loc='upper right')
    plt.show()
    
    
    
    t1 = time.time()
    print('Elapsed time is ', t1-t0)