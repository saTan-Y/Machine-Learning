# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 21:46:01 2017

@author: saTan-Y
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pprint import pprint
import time

if __name__ == '__main__':
    t0 = time.time()
    path = '10.Advertising.csv'
    
#    f = open(path)
#    x, y = [], []
#    ff = list(f)
#    for i, d in enumerate(ff):
#        if i == 0:
#            continue
#        d = d.strip()
#        if not d:
#            continue
#        temp = list(map(float, d.split(',')))
#        x.append(temp[1:-1])
#        y.append(temp[-1])

#    f = csv.reader(open(path))
#    for line in f:
#        print(line)

    data = pd.read_csv(path)
    x = data[['TV', 'Radio']]
    y = data['Sales']
    
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(9,12))
    plt.subplot(311)
    plt.plot(data['TV'], y, 'ro')
    plt.title('TV')
    plt.grid(True)
    plt.subplot(312)
    plt.plot(data['Radio'], y, 'g^')
    plt.title('Radio')
    plt.grid(True)
    plt.subplot(313)
    plt.plot(data['Newspaper'], y, 'b+')
    plt.title('Newspaper')
    plt.grid(True)
    plt.tight_layout()
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 1)
    lr = LinearRegression()
    model = lr.fit(x_train, y_train)
    print(model)
    print(lr.coef_, lr.intercept_)
    
    y_model = lr.predict(x_test)
    mse = np.average((y_model - y_test)**2)
    rmse = np.sqrt(mse)
    print('mse = %f, rmse = %f' % (mse, rmse))
    
    plt.figure()
    t = np.arange(len(x_test))
    plt.plot(t, y_test, 'r-', lw=2, label=u'原始')
    plt.plot(t, y_model, 'b-', lw=2, label=u'预测')
    plt.legend(loc = 'upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    t1 = time.time()
    print('Elapsed time is ', t1-t0)