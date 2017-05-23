#!/usr/bin/python
# -*- coding:utf-8 -*-

from time import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
import scipy as sp

if __name__ == '__main__':
    t0 = time()
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    np.random.seed(0)
    cov1 = np.diag((1, 2))
    N1 = 500
    N2 = 300
    N = N1 + N2
    x1 = np.random.multivariate_normal(mean=(3, 2), cov=cov1, size=N1)
    m = np.array(((1, 1), (1, 3)))
    x1 = x1.dot(m)
    x2 = np.random.multivariate_normal(mean=(-1, 10), cov=cov1, size=N2)
    x = np.vstack((x1, x2))
    y = np.array([0] * N1 + [1] * N2)
    n_components = 3

    colors = '#A0FFA0', '#2090E0', '#FF8080'
    cm = mpl.colors.ListedColormap(colors)
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
    x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]
    grid_test = np.stack((x1.flat, x2.flat), axis=1)

    plt.figure(figsize=(9, 9), facecolor='w')
    plt.suptitle(u'GMM/DPGMM比较', fontsize=23)

    ax = plt.subplot(211)
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
    gmm.fit(x)
    means = gmm.means_
    covs = gmm.covariances_
    print(means, covs)
    grid_plot = gmm.predict(grid_test)
    grid_plot = grid_plot.reshape(x1.shape)
    plt.pcolormesh(x1, x2, grid_plot, cmap=cm)
    plt.scatter(x[:, 0], x[:, 1], c=y, s=30, cmap=cm)

    clrs = list('grbmy')
    for i, (mean, cov) in enumerate(zip(means, covs)):
        eigen, vector = sp.linalg.eigh(cov)
        width, height = eigen[0], eigen[1]
        v = vector[0] / sp.linalg.norm(vector[0])
        theta = 180*np.arctan(v[1]/v[0])/np.pi
        e = Ellipse(xy=mean, width=width, height=height, angle=theta, alpha=0.7, color=clrs[i])
        ax.add_artist(e)
    ax1_min, ax1_max, ax2_min, ax2_max = plt.axis()
    plt.xlim(ax1_min, ax1_max)
    plt.ylim(ax2_min, ax2_max)
    plt.grid()
    plt.title(u'GMM', fontsize=20)

    ax = plt.subplot(212)
    dpgmm = BayesianGaussianMixture(n_components=n_components, covariance_type='full', max_iter=1000, n_init=5,
                                    weight_concentration_prior_type='dirichlet_process', weight_concentration_prior=10)
    dpgmm.fit(x)
    means = dpgmm.means_
    covs = dpgmm.covariances_
    print(means, covs)
    y_pred = dpgmm.predict(x)
    grid_plot = dpgmm.predict(grid_test)
    grid_plot = grid_plot.reshape(x1.shape)
    plt.pcolormesh(x1, x2, grid_plot, cmap=cm)
    plt.scatter(x[:, 0], x[:, 1], c = y, s = 30, cmap = cm)

    # clrs = list('grbmy')
    for i, (mean, cov) in enumerate(zip(means, covs)):
        if i not in y_pred:
            continue
        eigen, vector = sp.linalg.eigh(cov)
        width, height = eigen[0], eigen[1]
        v = vector[0] / sp.linalg.norm(vector[0])
        theta = 180 * np.arctan(v[1] / v[0]) / np.pi
        e = Ellipse(xy=mean, width=width, height=height, angle=theta, alpha=0.7, color=clrs[i])
        ax.add_artist(e)
    ax1_min, ax1_max, ax2_min, ax2_max = plt.axis()
    plt.xlim(ax1_min, ax1_max)
    plt.ylim(ax2_min, ax2_max)
    plt.grid()
    plt.title(u'DPGMM', fontsize=20)

    plt.tight_layout()
    plt.show()

