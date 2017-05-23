#!/usr/bin/python
# -*- coding:utf-8 -*-

from time import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
import warnings

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    np.random.seed(0)

    N1, N2 = 990, 10
    N = N1 + N2
    x1 = 3 * np.random.randn(N1, 2)
    x2 = 0.5 * np.random.randn(N2, 2) + (4, 4)
    x = np.vstack((x1, x2))
    y = np.zeros(N)
    y[N1:] = 1

    s = np.ones(N) * 30
    s[:N1] = 10

    model_lists = [svm.SVC(C=1, kernel='linear'),
                   svm.SVC(C=1, kernel='linear', class_weight={0:1, 1:10}),
                   svm.SVC(C=0.8, kernel='rbf', gamma=0.5, class_weight={0:1, 1:10}),
                   svm.SVC(C=0.8, kernel='rbf', gamma=0.5, class_weight={0:1, 1:20})]
    titles = ['linear', 'linear with weight', 'rbf with weigh1', 'rbf with weight2']
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
    xx, yy = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
    grid_test = np.stack((xx.flat, yy.flat), axis=1)

    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r'])
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    for i, model in enumerate(model_lists):
        model.fit(x, y)
        y_pred = model.predict(x)
        print('正确率', accuracy_score(y, y_pred))
        print('准确率', precision_score(y, y_pred))
        print('找回率', recall_score(y, y_pred))
        print('f1', f1_score(y, y_pred))
        grid_plot = model.predict(grid_test)
        grid_plot = grid_plot.reshape(xx.shape)
        plt.subplot(2, 2, i+1)
        plt.pcolormesh(xx, yy, grid_plot, cmap=cm_light)
        plt.scatter(x[:, 0], x[:, 1], c=y, s=s, cmap=cm_dark, edgecolors='k', alpha=0.8)
        plt.xlim(x1_min-0.2, x1_max+0.2)
        plt.ylim(x2_min-0.2, x2_max+0.2)
        plt.title(titles[i])
    plt.show()
