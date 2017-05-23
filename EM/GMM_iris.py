#!/usr/bin/python
# -*- coding:utf-8 -*-

from time import time
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import pairwise_distances_argmin
import matplotlib as mpl
import matplotlib.pyplot as plt

if __name__ == '__main__':
    t0 = time()
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    features = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'
    path = '..\\10.Regression\\10.iris.data'
    df = pd.read_csv(path, header=None)
    df[4] = df[4].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
    x_all, y = df.values[:, :4], df.values[:, -1]

    n_components = 3
    feature_list = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    plt.figure(figsize=(10, 9), facecolor='#FFFFFF')
    for i, item in enumerate(feature_list):
        x = x_all[:, item]
        means = np.array([np.mean(x[y==j], axis=0) for j in range(3)])
        print('实际均值', means)
        model = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
        model.fit(x)
        print(model.means_)
        y_pred = model.predict(x)
        order = pairwise_distances_argmin(means, model.means_, axis=1)
        print(order)

        n_types = 3
        n_samples = y.size
        temp = np.empty((n_types, n_samples), dtype=bool)
        for k in range(3):
            temp[k] = y_pred == order[k]
        for k in range(3):
            y_pred[temp[k]] = k
        print('acc rate', accuracy_score(y, y_pred))

        cm_light = mpl.colors.ListedColormap(['#FF8080', '#77E0A0', '#A0A0FF'])
        cm_dark = mpl.colors.ListedColormap(['r', 'g', '#6060FF'])
        x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
        x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
        xx, yy = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
        grid_test = np.stack((xx.flat, yy.flat), axis=1)
        grid_plot = model.predict(grid_test)
        temp = np.empty((n_types, grid_plot.size), dtype=bool)
        for k in range(3):
            temp[k] = grid_plot == order[k]
        for k in range(3):
            grid_plot[temp[k]] = k
        # print('acc rate', accuracy_score(y, y_pred))
        grid_plot = grid_plot.reshape(xx.shape)
        plt.subplot(2, 3, i+1)
        plt.pcolormesh(xx, yy, grid_plot, cmap=cm_light)
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cm_dark, s=50, marker='o', edgecolors='k')
        plt.xlim(x1_min-0.2, x1_max+0.2)
        plt.ylim(x2_min - 0.2, x2_max + 0.2)
        plt.grid()
        plt.xlabel(features[item[0]])
        plt.ylabel(features[item[1]])
        plt.text(0.9*x1_min+0.1*x1_max, 0.1*x2_min+0.9*x2_max, ('准确率：%.2f' % accuracy_score(y, y_pred)))
    plt.suptitle('sd')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    print('Elapsed time is', time()-t0)
    plt.show()


