#!/usr/bin/python
# -*- coding:utf-8 -*-

from time import time
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
import matplotlib.colors
from sklearn.model_selection import train_test_split

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    t0 = time()
    df = pd.read_csv('..\\20.EM\\20.HeightWeight.csv')
    print(df.columns)
    x, y = df.values[:, 1:], df.values[:, 0]
    # x = df[['Height(cm)', 'Weight(kg)']]
    # y = df['Sex']
    # print(df.values)
    # print(np.array(x))
    x_test, x_train, y_test, y_train = train_test_split(x, y, train_size=0.6, random_state=0)
    gmm = GaussianMixture(n_components=2, covariance_type='full', tol=1e-6, random_state=0)
    gmm.fit(x_test)
    y_test_pred = gmm.predict(x_test)
    y_train_pred = gmm.predict(x_train)
    if gmm.means_[0][0] > gmm.means_[1][0]:
        y_train_pred = np.where(y_train_pred == 1, 0, 1)
        y_test_pred = np.where(y_test_pred == 1, 0, 1)
    # print(accuracy_score(y_test, y_test_pred))
    # print(accuracy_score(y_train, y_train_pred))

    word1 = u'训练集准确率：%.3f' % accuracy_score(y_train, y_train_pred)
    word2 = u'预测集准确率：%.3f' % accuracy_score(y_test, y_test_pred)

    cm_light = mpl.colors.ListedColormap(['#FF8080', '#77E0A0'])
    cm_dark = mpl.colors.ListedColormap(['r', 'g'])

    x1_min, x2_min = np.min(np.array(x), axis=0)-5
    x1_max, x2_max = np.max(np.array(x), axis=0)+5
    x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]
    temp = np.stack((x1.flat, x2.flat), axis=1)
    y_plot = gmm.predict(temp)
    y_plot = y_plot.reshape(x1.shape)
    if gmm.means_[0][0] > gmm.means_[1][0]:
        y_plot = np.where(y_plot == 1, 0, 1)

    plt.figure(figsize=(9, 7), facecolor='w')
    plt.pcolormesh(x1, x2, y_plot, cmap=cm_light)
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_dark, s=50, edgecolors='k', marker='o')
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_dark, s=50, edgecolors='k', marker='^')

    p = gmm.predict_proba(temp)
    p = p[:, 0].reshape(x1.shape)
    line = plt.contour(x1, x2, p, levels=(0.2, 0.5, 0.8), colors=list('rgb'), lw=2)
    plt.clabel(line, fontsize=15, fmt='%.1f')
    a1, a2, b1, b2 = plt.axis()
    loc1 = 0.9*a1 + 0.1*a2
    loc2 = 0.1*b1 + 0.9*b2
    plt.text(loc1, loc2, word1, fontsize=15)
    loc2 = 0.15 * b1 + 0.85 * b2
    plt.text(loc1, loc2, word2, fontsize=15)
    plt.title(u'EM算法估算GMM的参数', fontsize=20)
    plt.xlabel('heights')
    plt.ylabel('weights')
    plt.tight_layout()
    print('Elapsed time is', time()-t0)

    plt.show()



