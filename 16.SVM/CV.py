#!/usr/bin/python
# -*- coding:utf-8 -*

from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
    np.random.seed(0)
    N = 100
    x = np.sort(np.random.uniform(0, 6, N), axis=0)
    y = 2 * np.sin(x) + 0.1*np.random.randn(N)
    x = x.reshape(-1, 1)
    print(x, y)

    model_rbf = svm.SVR(kernel='rbf')
    c = np.logspace(-2, 2, 10)
    gamma = np.logspace(-2, 2, 10)
    params = {'C':c, 'gamma':gamma}
    model = GridSearchCV(model_rbf, param_grid=params)
    model.fit(x, y)


    x_test = np.linspace(x.min(), 1.1*x.max(), 100).reshape(-1, 1)
    y_rbf = model.predict(x_test)

    plt.figure(facecolor='w')
    plt.plot(x_test, y_rbf, 'r-', lw=2, label='rbf')
    plt.scatter(x[model.best_estimator_.support_], y[model.best_estimator_.support_], s=50, edgecolors='k', c='r', marker='*', label='support vector')
    plt.plot(x, y, 'mo', markersize=6)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()