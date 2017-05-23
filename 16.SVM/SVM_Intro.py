#!/usr/bin/python
# -*- coding:utf-8 -*-

from time import time
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


if __name__ == '__main__':
    t0 = time()

    df = pd.read_csv('..\\10.Regression\\10.iris.data', header=None)
    df[4] = df[4].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}).astype(int)
    x, y = df.values[:,:4], df.values[:, 4]
    x1 = x[:, :2]
    x_train, x_test, y_train, y_test = train_test_split(x1, y, test_size=0.4)

    svvm = svm.SVC(C=1.0, kernel='rbf', gamma=20, decision_function_shape='ovr')
    svvm.fit(x_train, y_train.ravel())
    print(svvm.score(x_train, y_train))
    print(svvm.score(x_test, y_test))
    y_pred_train = svvm.predict(x_train)
    y_pred_test = svvm.predict(x_test)
    rate1 = accuracy_score(y_train, y_pred_train)
    rate2 = accuracy_score(y_test, y_pred_test)
    print(rate1)
    print(rate2)

    x1_min, x1_max = x1[:, 0].min(), x1[:, 0].max()
    x2_min, x2_max = x1[:, 1].min(), x1[:, 1].max()
    N = 200
    t1, t2 = np.linspace(x1_min, x1_max, N), np.linspace(x2_min, x2_max, N)
    xx, yy = np.meshgrid(t1, t2)
    x11, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
    grid_test = np.stack((xx.flat, yy.flat), axis=1)
    grid_plot = svvm.predict(grid_test)
    grid_plot = grid_plot.reshape(xx.shape)

    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])


    plt.figure(facecolor='w')
    plt.pcolormesh(xx, yy, grid_plot, cmap=cm_light)
    plt.scatter(x1[:, 0], x1[:, 1], s=40, c=y, edgecolors='k', cmap=cm_dark)
    plt.scatter(x_test[:, 0], x_test[:, 1], s=100, marker='o', facecolors='none', zorder=10)
    plt.xlim(x1_min-0.5, x1_max+0.5)
    plt.ylim(x2_min-0.5, x2_max+0.5)
    plt.xlabel('sepal length', fontsize=12)
    plt.ylabel('sepal width', fontsize=12)
    # plt.tight_layout()
    plt.grid()
    plt.title('iris')
    plt.show()
    print('Elapsed time is', time()-t0)