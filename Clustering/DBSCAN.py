#!/usr/bin/python
# -*- coding:utf-8 -*-

from time import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import sklearn.datasets as ds

if __name__ == '__main__':
    t0 = time()
    N = 1000
    centers = [(1, 2), (1, -1), (-1, -1), (-1, 1)]
    data, y = ds.make_blobs(N, n_features=2, centers=centers, cluster_std=[0.5, 0.25, 0.7, 0.5], random_state=1)
    params = [(0.2, 5), (0.2, 10), (0.2, 15), (0.3, 5), (0.3, 10), (0.3, 15)]

    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(12, 8), facecolor='w')
    plt.suptitle(u'DBSCAN聚类', fontsize=20)
    for i, item in enumerate(params):
        model = DBSCAN(eps=item[0], min_samples=item[1])
        model.fit(data)
        y_pred = model.labels_

        core_indices = np.zeros_like(y_pred, dtype=bool)
        core_indices[model.core_sample_indices_] = True

        n = np.unique(y_pred).size - np.where(-1 in y_pred, 1, 0)
        print(np.unique(y_pred), '聚类簇的个数为：', n)

        plt.subplot(2, 3, i+1)
        clrs = plt.cm.Spectral(np.linspace(0, 0.8, np.unique(y_pred).size))
        for k, clr in zip(y_pred, clrs):
            if k == -1:
                plt.scatter(data[:, 0][y_pred==k], data[:, 1][y_pred==k], c='k', s=20)
                continue
            plt.scatter(data[:, 0][y_pred==k], data[:, 1][y_pred==k], c=clr, s=30, edgecolors='k')

        plt.title('$\epsilon$ = %1f, m = %d, num = %d' % (item[0], item[1], n))
        x1_min, x1_max = data[:, 0].min(), data[:, 0].max()
        x2_min, x2_max = data[:, 1].min(), data[:, 1].max()
        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)
        plt.grid()
    plt.tight_layout()
    plt.subplots_adjust(top=0.91)
    plt.show()



