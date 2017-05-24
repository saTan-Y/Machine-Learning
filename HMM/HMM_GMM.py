#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from hmmlearn import hmm
import warnings
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    np.random.seed(0)
    n, n_samples = 5, 500
    pi = np.random.rand(n)
    pi /= pi.sum()
    print('初始概率：', pi)

    A = np.random.rand(n, n)
    sign = np.zeros((n, n))
    temp = np.array([4, 0, 1, 2, 3, 4, 0])
    # print(temp)
    for i in temp[1:-2]:
        sign[i, i-1] = sign[i, i+1] = True
    # print(sign)
    A[sign == 1] = 0
    for i in range(n):
        A[i] /= A[i].sum()
    print('转移概率：', A)

    means = np.array(((30, 30, 30), (0, 50, 20), (-25, 30, 10), (-15, 0, 25), (15, 0, 40)), dtype=np.float)
    # means = np.random.rand(n, 3)
    for i in range(n):
        means[i] /= np.sqrt(np.sum(means[i]**2))
    print('均值：', means)

    covars = np.empty((n, 3, 3))
    for i in range(n):
        covars[i] = np.diag(np.random.rand(3) * 0.03 + 0.01)
    print('方差：', covars)

    model1 = hmm.GaussianHMM(n_components=5, covariance_type='full')
    model1.startprob_ = pi
    model1.transmat_ = A
    model1.means_ = means
    model1.covars_ = covars
    samples, labels = model1.sample(n_samples=n_samples, random_state=0)

    model2 = hmm.GaussianHMM(n_components=5, covariance_type='full')
    model2.fit(samples)
    # print(samples)
    y = model2.predict(samples)
    print('估计概率初始：', model2.startprob_)
    print('估计概率转移：', model2.transmat_)
    print('估计均值：', model2.means_)
    print('估计方差：', model2.covars_)

    order = pairwise_distances_argmin(means, model2.means_, metric='euclidean')
    print(order)
    pi_est = model2.startprob_[order]
    A_est = model2.transmat_[order]
    A_est = A_est[:, order]
    means_est = model2.means_[order]
    covars_est = model2.covars_[order]
    change = np.empty((n, n_samples), dtype = np.bool)
    for i in range(n):
        change[i] = y == order[i]
    for i in range(n):
        y[change[i]] = i
    print('估计概率初始2：', pi_est)
    print('估计概率转移2：', A_est)
    print('估计均值2：', means_est)
    print('估计方差2：', covars_est)
    accu_rate = accuracy_score(labels, y)
    print('准确率：', accu_rate)

    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(10, 8), facecolor='w')
    ax = fig.add_subplot(111, projection='3d')
    colors = mpl.cm.Spectral(np.arange(0, 1, n))
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], s=50, c=labels, cmap=mpl.cm.Spectral, marker='o', label='u观测值')
    plt.plot(samples[:, 0], samples[:, 1], samples[:, 2], color='#A07070', lw=0.1, )
    ax.scatter(means[:, 0], means[:, 1], means[:, 2], s=200, c=colors, edgecolor='r', marker='*', label='u中心')

    x_min, y_min, z_min = samples.min(axis=0)
    x_max, y_max, z_max = samples.max(axis=0)
    ax.set_xlim((x_min-0.1, x_max+0.1))
    ax.set_ylim((y_min - 0.1, y_max + 0.1))
    ax.set_zlim((z_min - 0.1, z_max + 0.1))
    plt.legend(loc='upper left')
    plt.tight_layout(1)
    plt.grid()
    plt.title('GMMHMM', fontsize=18)
    plt.show()


