#!/usr/bin/python
# -*- coding:utf-8 -*-

from time import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from sklearn.cluster import KMeans

def restore_image(cb, cluster, shape):
    x1, x2, x3 = shape
    ig = np.empty((x1, x2, 3))
    k = 0
    for i in range(x1):
        for j in range(x2):
            ig[i, j] = cb[cluster[k]]
            k += 1
    return ig

def show_scatter(x):
    N = 10
    print(u'原始数据', x)
    fig = plt.figure(1, facecolor='w')
    H, edge = np.histogramdd(x, bins=(N, N, N), range=((0, 1), (0, 1), (0, 1)))
    H /= H.max()
    xx = y = z = np.arange(N)
    t = np.meshgrid(xx, y, z)

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(t[0], t[1], t[2], s=H*100, c='r', marker='o', depthshade=True)
    ax.set_xlabel(u'红色分量')
    ax.set_ylabel(u'绿色分量')
    ax.set_zlabel(u'蓝色分量')
    plt.title(u'图像颜色三维频数分布')

    plt.figure(2, facecolor='w')
    H = H[H>0]
    H = np.sort(H)[::-1]
    t = np.arange(len(H))
    plt.plot(t, H, 'r-', t, H, 'go', lw=2)
    plt.title(u'图像颜色频数分布', fontsize=18)
    plt.show()

if __name__ == "__main__":
    t0 = time()
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    path = '18.Lena.png'
    ig1 = Image.open(path)
    ig2 = np.array(ig1).astype(dtype='int') / 255
    ig3 = ig2[:, :, :3]
    ig4 = ig3.reshape(-1, 3)
    show_scatter(ig4)

    num_color = 7
    model = KMeans(n_clusters=num_color)
    N = len(ig4)
    temp = np.random.randint(0, N, size=1000)
    y1 = model.fit_predict(ig4[temp])
    y = model.fit_predict(ig4)
    print('聚类结果：\n', y)
    print('聚类中心：\n', model.cluster_centers_)

    plt.figure(facecolor='w')
    plt.subplot(121)
    plt.title(u'原始图片', fontsize=18)
    plt.imshow(ig2)
    plt.axis('off')

    plt.subplot(122)
    plt.title(u'ggg', fontsize=18)
    plt.imshow(restore_image(model.cluster_centers_, y, ig2.shape))
    plt.axis('off')
    print('Elapsed time is', time()-t0)
    plt.show()