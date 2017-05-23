#!/usr/bin/python
# -*- coding:utf-8 -*-

from time import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    t0 = time()
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    np.random.seed(0)

    type = ['spherical', 'diag', 'tied', 'full']
    means1 = (1, 2)
    means2 = (-1, 10)
    cov = np.diag((1, 2))
    N1, N2 = 500, 300
    x1 = np.random.multivariate_normal(means1, cov, size=N1)
    x2 = np.random.multivariate_normal(means2, cov, size=N2)
    x = np.vstack((x1, x2))
    y = np.array([1]*N1 + [0]*N2)

    err = np.zeros(len(type))
    bic = np.zeros(len(type))

    for i, item in enumerate(type):
        model = GaussianMixture(n_components=2, covariance_type=item, random_state=0)
        model.fit(x, y)
        err[i] = 1 - accuracy_score(y, model.predict(x))
        bic[i] = model.bic(x)
    print(err, bic)

    xpos = np.arange(4)
    ax = plt.axes()
    b1 = ax.bar(xpos-0.3, err, width=0.3, color='r')
    b2 = ax.twinx().bar(xpos, bic, width=0.3, color='g')
    plt.grid()
    plt.ylim(bic.min()-100, bic.max()+100)
    plt.xticks(xpos, type)
    plt.legend((b1[0], b2[0]), ('err', 'bic'))
    plt.show()

    best = bic.argmin()
    model = GaussianMixture(2, covariance_type=type[best], random_state=0)
    model.fit(x, y)
    y_pred = model.predict(x)
    print(model.means_, model.covariances_)

    cm_light = mpl.colors.ListedColormap(['#FF8080', '#77E0A0'])
    cm_dark = mpl.colors.ListedColormap(['r', 'g'])

    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
    xx, yy = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
    grid_test = np.stack((xx.flat, yy.flat), axis=1)
    grid_plot = model.predict(grid_test)
    grid_plot = grid_plot.reshape(xx.shape)

    plt.figure(facecolor='w')
    plt.pcolormesh(xx, yy, grid_plot, cmap=cm_light)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cm_dark, edgecolors='k', s=50)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid()
    plt.title('sdfjs')
    plt.show()
