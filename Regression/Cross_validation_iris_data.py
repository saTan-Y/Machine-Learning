# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 12:40:49 2017

@author: saTan-Y
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV
import time

if __name__ == '__main__':
    t0 = time.time()
    path = '10.Advertising.csv'
    
    data = pd.read_csv(path)
    x = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']
    
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
                
    model = Ridge()
    alpha = np.logspace(-3, 2, 10)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=1)
    lasso_model = GridSearchCV(model, param_grid = {'alpha':alpha}, cv=3)
    lasso_model.fit(x_train, y_train)
    print('best parameter is ', lasso_model.best_params_)
    
    y_model = lasso_model.predict(x_test)
    print(lasso_model.score(x_test, y_test))
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
