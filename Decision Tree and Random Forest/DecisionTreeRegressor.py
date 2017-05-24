# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 14:29:13 2017

@author: saTan-Y
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

if __name__ == '__main__':
    N = 100
    x = np.random.rand(N) * 6 - 3
    x.sort()
    y = np.sin(x) + np.random.randn(N)*0.05
    x.reshape(-1, 1)
    x = x[:, None]
    
    model = DecisionTreeRegressor(criterion='mse', max_depth=9)
    model = model.fit(x, y)
    x_test = np.linspace(-3, 3, 50).reshape(-1,1)
    y_pred = model.predict(x_test)
    
    plt.figure()
    plt.plot(x, y, 'r*', lw=2, label='test')
    plt.plot(x_test, y_pred, 'g-', lw=2, label='prediction')
    plt.legend(loc='upper left')
    plt.show()
    
    depth = [2, 4, 6, 8, 10]
    colors = 'ymbcg'
    for i, d in enumerate(depth):
        model = DecisionTreeRegressor(criterion='mse', max_depth=d)
        model.fit(x, y)
        y_pred = model.predict(x_test)
        plt.plot(x_test, y_pred, color=colors[i], lw=2, label='Depth=%d' % d)
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()