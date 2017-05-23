#!/usr/bin/python
# -*- coding:utf-8 -*-

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import  pairwise_distances_argmin
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    t0 = time.time()
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    # choice = 'diy'
    choice = 'sklearn'
    np.random.seed(0)
    mu1_real, mu2_real = (0, 0, 0), (2, 2, 2)
    sigma1_real, sigma2_real = 2 * np.identity(3), np.identity(3) #np.array(((3, 0.5, 2), (1, 1, 1), (0, 0, 0.5)))
    N1, N2 = 500, 200
    # print(np.linalg.det(sigma2_real))
    data1 = np.random.multivariate_normal(mu1_real, sigma1_real, N1)
    data2 = np.random.multivariate_normal(mu2_real, sigma2_real, N2)
    data = np.vstack((data1, data2))
    y_real = np.array([True] * N1 + [False] * N2)
    # print(y_real)

    if choice == 'sklearn':
        gmm = GaussianMixture(2, covariance_type='full', tol=1e-6, max_iter=1000)
        gmm.fit(data)
        print(u'类别概率\t', gmm.weights_)
        print(u'均值\n', gmm.means_)
        print(u'方差\n', gmm.covariances_)
        mu1, mu2 = gmm.means_
        sigma1, sigma2 = gmm.covariances_
    else:
        num_iter = 300
        n, d = data.shape
        mu1 = data.min(axis=0)
        mu2 = data.max(axis=0)
        print(mu1, mu2)
        sigma1 = np.identity(3)
        sigma2 = 2 * np.identity(3)

        for i in range(num_iter):
            distribution1 = multivariate_normal(mu1, sigma1)
            distribution2 = multivariate_normal(mu2, sigma2)
            tau1 = distribution1.pdf(data)
            tau2 = distribution2.pdf(data)
            gamma1 = tau1 / (tau1 + tau2)
            gamma2 = tau2 / (tau1 + tau2)

            N_k1 = np.sum(gamma1)
            N_k2 = np.sum(gamma2)
            mu1 = np.dot(gamma1, data) / N_k1
            mu2 = np.dot(gamma2, data) / N_k2
            sigma2 = np.dot(gamma1 * (data - mu1).T, (data - mu1)) / N_k1
            sigma1 = np.dot(gamma2 * (data - mu2).T, (data - mu2)) / N_k2
            pi1 = np.sum(gamma1) / n
            pi2 = np.sum(gamma2) / n
        print(u'类别概率\t', [pi1, pi2])
        print(u'均值\n', mu1, mu2)
        print(u'方差\n', sigma1, '\n', sigma2)

    distribution1 = multivariate_normal(mu1, sigma1)
    distribution2 = multivariate_normal(mu2, sigma2)
    tau1 = distribution1.pdf(data)
    tau2 = distribution2.pdf(data)
    order = pairwise_distances_argmin([mu1_real, mu2_real], [mu1, mu2])
    if order[0] == 0:
        c = tau1 > tau2
    else:
        c = tau1 < tau2
    c2 = ~c

    print('accuracy rate', accuracy_score(y_real, c))
    fig = plt.figure(figsize=(13, 7), facecolor='w')
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(data[:, 0], data[:, 1], data[:, 2], s=40, marker='o', c='r', depthshade=True)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title(u'原始数据', fontsize=18)

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(data[c, 0], data[c, 1], data[c, 2], s=40, marker='o', c='r', depthshade=True)
    ax2.scatter(data[c2, 0], data[c2, 1], data[c2, 2], s=40, marker='^', c='g', depthshade=True)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.set_title(u'EM算法分类', fontsize=18)
    plt.suptitle('EM', fontsize=25)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    print('Elapsed time is', time.time()-t0)
    plt.show()

