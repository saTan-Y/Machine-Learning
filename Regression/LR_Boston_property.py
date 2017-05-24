# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 21:01:10 2017

@author: saTan-Y
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import sklearn.datasets
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

def not_empty(s):
    return s != ''
    
if __name__ == '__main__':
    t0 = time.time()
    boston = sklearn.datasets.load_boston()
    b = boston.target
    df = pd.read_csv('10.housing.data', header=None)
    a = df.values
    m = len(a)
    data = np.zeros((m, 14))
    for i, item in enumerate(a):
#        print(item[0])
        item = item[0].split(' ')
        temp = list(map(float, filter(not_empty, item)))
        data[i] = temp
    x, y = np.split(data, (13,), axis=1)
    
    model = Pipeline([('ss', StandardScaler()), ('poly', PolynomialFeatures(degree=3, include_bias=True)), ('ln', ElasticNetCV(alphas=np.logspace(-3,2,10), 
                                            l1_ratio=[0.05, 0.1, 0.2, 0.4, 0.5, 0.7, 0.9, 0.95], fit_intercept=False, max_iter=3, cv=5))])
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)
    model.fit(x_train, y_train.ravel())
    linear = model.get_params('ln')['ln']
    print(u'超参数为', linear.alpha_)
    print(u'l1 ratio is ', linear.l1_ratio_)
    
    y_pred = model.predict(x_test)
    R2 = model.score(x_test, y_test)
    mse = mean_squared_error(y_test, y_pred)
    print('R2 is', R2)
    print('mse is ', mse)
    
    t = np.arange(len(y_test))
    plt.figure()
    plt.plot(t, y_test, 'r-', lw=2, alpha=0.7, label='real value')
    plt.plot(t, y_pred, 'g-', lw=2, alpha=0.7, label='predicted value')
    plt.legend(loc='upper right')
    plt.title('prediction of Boston property')
    plt.grid()
    plt.tight_layout()
    plt.show()
    
    t1 = time.time()
    print('Elapsed time is ', t1-t0)
    

