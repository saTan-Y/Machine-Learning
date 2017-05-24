#!/usr/bin/python
# -*- coding:utf-8 -*-

from time import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from sklearn.metrics import euclidean_distances
import sklearn.datasets as ds

if __name__ == '__main__':
    to = time()
    N = 1000
    centers = [(1, 2), (1, -1), (-1, -1), (-1, 1)]
    data, y = ds.make_blobs(N, n_features=2, centers=centers, cluster_std=[0.5, 0.25, 0.7, 0.5], random_state=1)

    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    dist = euclidean_distances(data)
    d = np.median(dist)
    plt.figure(facecolor='w')
    print(d)

    for i, item in enumerate(np.linspace(0.1, 0.4, 4)):
        bw = item * d
        model = MeanShift(bandwidth=bw, bin_seeding=True)
        model.fit(data)
        center = model.cluster_centers_
        y_pred = model.labels_
        n_labels = np.unique(y_pred).size

        print(bw, n_labels)
        plt.subplot(2, 2, i+1)
        clrs = []
        for c in np.linspace(16711680, 255, n_labels):
            clrs.append('#%06x' % int(c))
        for j, clr in enumerate(clrs):
            plt.scatter(data[:, 0][y_pred==j], data[:, 1][y_pred==j], s=40, c=clr)
        plt.scatter(center[:, 0], center[:, 1], c=clrs, marker='*', s=150)
        plt.title(u'带宽：%.2f，聚类簇的个数为：%d' % (bw, n_labels))
        plt.grid()
    plt.tight_layout()
    plt.show()

