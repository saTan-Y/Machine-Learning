#!/usr/bin/python
# -*- coding:utf-8 -*-

from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

if __name__ == '__main__':
    np.random.seed(0)
    N = 100
    x = np.sort(np.random.uniform(0, 6, N), axis=0)
    y = 2 * np.sin(x) + 0.1*np.random.randn(N)
    x = x.reshape(-1, 1)
    print(x, y)

    model_rbf = svm.SVR(C=100, kernel='rbf', gamma=0.2)
    model_rbf.fit(x, y)
    model_linear = svm.SVR(C=100, kernel='linear')
    model_linear.fit(x, y)
    model_poly = svm.SVR(C=100, kernel='poly', degree=3)
    model_poly.fit(x, y)

    x_test = np.linspace(x.min(), 1.1*x.max(), 100).reshape(-1, 1)
    y_rbf = model_rbf.predict(x_test)
    y_linear = model_linear.predict(x_test)
    y_poly = model_poly.predict(x_test)

    plt.figure(facecolor='w')
    plt.plot(x_test, y_rbf, 'r-', lw=2, label='rbf')
    plt.plot(x_test, y_linear, 'g-', lw=2, label='linear')
    plt.plot(x_test, y_poly, 'b-', lw=2, label='poly')
    plt.scatter(x[model_rbf.support_], y[model_rbf.support_], s=50, edgecolors='k', c='r', marker='*', label='support vector')
    plt.plot(x, y, 'mo', markersize=6)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()