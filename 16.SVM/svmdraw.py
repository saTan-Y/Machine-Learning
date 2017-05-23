#!/usr/bin/python
# -*- coding:utf-8 -*-

import time
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    t0 = time.time()
    df = pd.read_csv('..\\16.SVM\\16.bipartition.txt', header=None, delimiter='\t')
    x, y = df.values[:, :-1], df.values[:, -1]
    y[y == 0] = -1

    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
    t1, t2 = np.linspace(x1_min, x1_max, 500), np.linspace(x2_min, x2_max, 500)
    x1, x2 = np.meshgrid(t1, t2)
    temp = np.stack((x1.flat, x2.flat), axis=1)
    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FFA0A0'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r'])
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    model_list = (('linear', 0.1), ('linear', 0.5), ('linear', 1), ('linear', 2),
                  ('rbf', 1, 0.1), ('rbf', 1, 1), ('rbf', 1, 10), ('rbf', 1, 100),
                  ('rbf', 5, 0.1), ('rbf', 5, 1), ('rbf', 5, 10), ('rbf', 5, 100))
    plt.figure(figsize=(14,10), facecolor='w')
    for i, item in enumerate(model_list):
        model = svm.SVC(C=item[1], kernel=item[0], decision_function_shape='ovr', random_state=1)
        if item[0] == 'rbf':
            model.gamma = item[2]
            title = u'高斯核, C=%.2f, gamma=%.2f' % (item[1], item[2])
        else:
            title = u'线性核, C=%.2f' % (item[1])
        model.fit(x, y)
        y_pred = model.predict(x)
        y_plot = model.predict(temp)
        y_plot = y_plot.reshape(x1.shape)
        y_con = model.decision_function(temp)
        y_con = y_con.reshape(x1.shape)
        print(title)
        print(u'准确率', accuracy_score(y, y_pred))
        print(u'支撑向量个数', model.n_support_)
        print(u'支撑向量系数', model.dual_coef_)
        print(u'支撑向量', model.support_)
        plt.subplot(3, 4, i+1)
        plt.pcolormesh(x1, x2, y_plot, cmap=cm_light)
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cm_dark, edgecolors='k')
        plt.scatter(x[model.support_, 0], x[model.support_, 1], s=100, facecolors='none', edgecolors='k')
        plt.contour(x1, x2, y_con, colors=list('kmrkb'), linestyles=['-', '--', '-.', '-', '-.'], linewidths=[1,2,2,3,1])
        plt.title(title)
        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)
        plt.tight_layout(1.4)

    plt.show()

    t1 = time.time()
    print('Elapsed time is', t1-t0)


