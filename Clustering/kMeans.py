#!/usr/bin/python
# -*- coding:utf-8 -*-

from time import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.cluster import KMeans
import sklearn.datasets as ds

if __name__ == '__main__':
    t0 = time()
    N = 800
    n_features = 2
    centers = 4
    data1, y1 = ds.make_blobs(n_samples=N, n_features=n_features, centers=centers, random_state=1)
    data2, y2 = ds.make_blobs(n_samples=N, n_features=n_features, centers=centers, cluster_std=(1, 2.5, 0.5, 2), random_state=1)
    data3 = np.vstack((data1[y1==0][:50], data1[y1==1][:100], data1[y1==2][:150], data1[y1==3][:200]))
    y3 = np.array([0]*50 + [1]*100 + [2]*150 + [3]*200)
    # y3 = np.array([0] * 100 + [1] * 50 + [2] * 20 + [3] * 5)

    k = 4
    model = KMeans(n_clusters=k, init='k-means++')
    y_p1 = model.fit_predict(data1)
    y_p2 = model.fit_predict(data2)
    y_p3 = model.fit_predict(data3)

    r = np.array([[1, 1], [1, 3]])
    datar = data1.dot(r)
    yr = model.fit_predict(datar)

    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    cm = mpl.colors.ListedColormap(list('rgbm'))

    plt.figure(figsize=(9, 10), facecolor='w')
    plt.subplot(4, 2, 1)
    x1_min, x1_max = data1[:, 0].min(), data1[:, 0].max()
    x2_min, x2_max = data1[:, 1].min(), data1[:, 1].max()
    plt.scatter(data1[:, 0], data1[:, 1], s=30, c=y1, cmap=cm, edgecolors='none')
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title(u'原始数据')
    plt.grid()
    # plt.show()

    plt.subplot(4, 2, 2)
    # x1_min, x1_max = data1[:, 0].min(), data1[:, 0].max()
    # x2_min, x2_max = data1[:, 1].min(), data1[:, 1].max()
    plt.scatter(data1[:, 0], data1[:, 1], s=30, c=y_p1, cmap=cm, edgecolors='none')
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title(u'KMeans++聚类')
    plt.grid()
    # plt.show()

    plt.subplot(4, 2, 3)
    xr1_min, xr1_max = datar[:, 0].min(), datar[:, 0].max()
    xr2_min, xr2_max = datar[:, 1].min(), datar[:, 1].max()
    plt.scatter(datar[:, 0], datar[:, 1], s=30, c=y1, cmap=cm, edgecolors='none')
    plt.xlim(xr1_min, xr1_max)
    plt.ylim(xr2_min, xr2_max)
    plt.title(u'旋转后数据')
    plt.grid()
    # plt.show()

    plt.subplot(4, 2, 4)
    # xr1_min, xr1_max = datar[:, 0].min(), datar[:, 0].max()
    # xr2_min, xr2_max = datar[:, 1].min(), datar[:, 1].max()
    plt.scatter(datar[:, 0], datar[:, 1], s=30, c=yr, cmap=cm, edgecolors='none')
    plt.xlim(xr1_min, xr1_max)
    plt.ylim(xr2_min, xr2_max)
    plt.title(u'旋转后KMeans++聚类')
    plt.grid()

    plt.subplot(4, 2, 5)
    x1_min, x1_max = data2[:, 0].min(), data2[:, 0].max()
    x2_min, x2_max = data2[:, 1].min(), data2[:, 1].max()
    plt.scatter(data2[:, 0], data2[:, 1], s=30, c=y2, cmap=cm, edgecolors='none')
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title(u'方差不相等数据')
    plt.grid()

    plt.subplot(4, 2, 6)
    # x1_min, x1_max = data2[:, 0].min(), data2[:, 0].max()
    # x2_min, x2_max = data2[:, 1].min(), data2[:, 1].max()
    plt.scatter(data2[:, 0], data2[:, 1], s=30, c=y_p2, cmap=cm, edgecolors='none')
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title(u'方差不相等KMeans++聚类')
    plt.grid()

    plt.subplot(4, 2, 7)
    x1_min, x1_max = data3[:, 0].min(), data3[:, 0].max()
    x2_min, x2_max = data3[:, 1].min(), data3[:, 1].max()
    plt.scatter(data3[:, 0], data3[:, 1], s=30, c=y3, cmap=cm, edgecolors='none')
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title(u'数量不相等数据')
    plt.grid()

    plt.subplot(4, 2, 8)
    # x1_min, x1_max = data3[:, 0].min(), data3[:, 0].max()
    # x2_min, x2_max = data3[:, 1].min(), data3[:, 1].max()
    plt.scatter(data3[:, 0], data3[:, 1], s=30, c=y_p3, cmap=cm, edgecolors='none')
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title(u'数量不相等KMeans++聚类')
    plt.grid()

    plt.tight_layout(2)
    plt.suptitle(u'数据分布对KMeans聚类的影响', fontsize=18)
    plt.subplots_adjust(top=0.92)
    print('Elapsed time is', time()-t0)
    plt.show()
