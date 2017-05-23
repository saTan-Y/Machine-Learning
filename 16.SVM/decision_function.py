#!/usr/bin/python
# -*- coding:utf-8 -*-

from time import time
import numpy as np
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import svm

if __name__ == '__main__':
    np.random.seed(0)
    N = 200
    x = np.empty((4*N, 2))
    means = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    sigmas = [np.eye(2), 2*np.eye(2), np.diag((1, 2)), np.array([[2, 1], [1, 2]])]
    for i in range(4):
        temp = stats.multivariate_normal(means[i], sigmas[i]*0.3)
        x[i*N: i*N+N, :] = temp.rvs(N)
    a = np.array((0, 1, 2, 3))
    y = np.tile(a.reshape(-1, 1), N).flatten()

    model = svm.SVC(C=1.0, kernel='rbf', gamma=40, decision_function_shape='ovr')
    model.fit(x, y)
    y_pred = model.predict(x)
    rate = accuracy_score(y, y_pred)
    print('accuracy rate is', rate)
    print(model.decision_function(x))

    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
    # x1_min, x2_min = np.min(x, axis=0)
    # x1_max, x2_max = np.max(x, axis=0)
    N = 200
    t1, t2 = np.linspace(x1_min, x1_max, N), np.linspace(x2_min, x2_max, N)
    # print(t2)
    xx, yy = np.meshgrid(t1, t2)
    grid_test = np.stack((xx.flat, yy.flat), axis=1)
    grid_plot = model.predict(grid_test)
    gird_plot = grid_plot.reshape(xx.shape)
    # print(grid_plot[44:100])

    # x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    # x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
    # N = 200
    # t1, t2 = np.linspace(x1_min, x1_max, N), np.linspace(x2_min, x2_max, N)
    # xx, yy = np.meshgrid(t1, t2)
    # x11, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
    # grid_test = np.stack((xx.flat, yy.flat), axis=1)
    # grid_plot = model.predict(grid_test)
    # grid_plot = grid_plot.reshape(xx.shape)

    cm_light = mpl.colors.ListedColormap(['#FF8080', '#A0FFA0', '#6060FF', '#F080F0'])
    cm_dark = mpl.colors.ListedColormap(['r', 'g', 'b', 'm'])
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    plt.figure(facecolor='w')
    plt.pcolormesh(xx, yy, grid_plot, cmap=cm_light)
    plt.scatter(x[:, 0], x[:, 1], s=40, c=y, cmap=cm_dark, edgecolors='k')
    plt.xlim(x1_min-0.1, x1_max+0.1)
    plt.ylim(x2_min-0.1, x2_max+0.1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('TITLE')
    plt.tight_layout()
    plt.grid()
    plt.show()

