#!/usr/bin/python
# -*- coding:utf-8 -*-

from time import  time
import numpy as np
import matplotlib as mpl
from sklearn import svm
import matplotlib.pyplot as plt

if __name__ == '__main__':
    t = np.linspace(-5, 5, 6)
    t1, t2 = np.meshgrid(t, t)
    x1 = np.stack((t1.flat, t2.flat), axis=1)
    x2 = x1 + (1, 1)
    x = np.concatenate((x1, x2))
    y = np.concatenate((-np.ones(len(x1)), np.ones(len(x1))))
    print(x.size, y.size)

    model = svm.SVC(C=1, kernel='rbf', gamma=5)
    model.fit(x, y)

    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
    xx, yy = np.mgrid[x1_min:x1_max:100j, x2_min:x2_max:100j]
    grid_test = np.stack((xx.flat, yy.flat), axis=1)
    grid_plot = model.predict(grid_test)
    grid_plot = grid_plot.reshape(xx.shape)

    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FFA0A0'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r'])

    plt.figure(facecolor='w')
    plt.pcolormesh(xx, yy, grid_plot, cmap=cm_light)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cm_dark, edgecolors='k')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('title')
    plt.show()