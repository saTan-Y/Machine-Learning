# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 16:04:24 2017

@author: saTan-Y
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

def xss(y, y_model):
    y = y.ravel()
    y_model = y_model.ravel()
    tss = ((y - y.mean())**2).sum()
    rss = ((y - y_model)**2).sum()
    ess = ((y_model - y.mean())**2).sum()
    R2 = 1 - rss/tss
        
    tss_list.append(tss)
    rss_list.append(rss)
    ess_list.append(ess)
    essrss_list.append(tss+rss)
    corcoe = np.corrcoef(y, y_model)
    return R2, corcoe
    
if __name__ == '__main__':
    t0 = time.time()
    np.random.seed(0)
    N = 9
    x = np.linspace(0, 6, N) + np.random.randn(N)
    x = np.sort(x)
    y = x**2 - 4*x -3 +np.random.randn(N)
    x.shape = -1,1
    y.shape = -1,1
    
    models = [Pipeline([('poly', PolynomialFeatures()), ('ln', LinearRegression(fit_intercept=False))]),
                      Pipeline([('poly', PolynomialFeatures()), ('ln', LassoCV(alphas=np.logspace(-3,2,50), fit_intercept=False))]),
                      Pipeline([('poly', PolynomialFeatures()), ('ln', RidgeCV(alphas=np.logspace(-3,2,50), fit_intercept=False))]),
                      Pipeline([('poly', PolynomialFeatures()), 
                                ('ln', ElasticNetCV(alphas=np.logspace(-3,2,50), l1_ratio=np.logspace(0.1,0.99,10), fit_intercept=False))])]
    
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    np.set_printoptions(suppress=True)
    
    order = np.arange(1, N, 1)
    m = order.size
    clrs = []  # 颜色
    for c in np.linspace(16711680, 255, m):
        clrs.append('#%06x' % int(c))
    line_width = np.linspace(5, 2, m)
    title = [u'线性回归', u'Ridge回归', u'LASSO', u'ElasticNet']
    tss_list = []
    rss_list = []
    ess_list = []
    essrss_list = []
    
    plt.figure(figsize=(18,12), facecolor='w')
    for t in range(4):
        model = models[t]
        plt.subplot(2, 2, t+1)
        plt.plot(x, y, 'ro', zorder=N)
        for i, d in enumerate(order):
            model.set_params(poly__degree=d)
            model.fit(x, y.ravel())
            lin = model.get_params('ln')['ln']
            output = u'%s: %d阶， 系数为' % (title[t], d)
            if hasattr(lin, 'alpha_'):
                idx = output.find(u'系数')
                output = output[:idx] + ('alpha = %6f' % lin.alpha_) + output[idx:]
            if hasattr(lin, 'l1_ratio_'):
                idx = output.find(u'系数')
                output = output[:idx] + ('l1_ratio = %6f' % lin.l1_ratio_) + output[idx:]
            print(output, lin.coef_.ravel())
            x_pred = np.linspace(x.min(), x.max(), 100)
            x_pred.shape = -1, 1
            y_pred = model.predict(x_pred)
            s = model.score(x_pred, y_pred)
            R2, corcoe = xss(y, model.predict(x))
            z = N - 1 if (d == 2) else 0
            label = u'%d阶，$R^2$=%.3f' % (d, s)
            if hasattr(lin, 'l1_ratio_'):
                label += u'，L1 ratio=%.2f' % lin.l1_ratio_
            plt.plot(x_pred, y_pred, color=clrs[i], lw=line_width[i], alpha=0.75, label=label, zorder=z)
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.title(title[t], fontsize=18)
        plt.xlabel('X', fontsize=16)
        plt.ylabel('Y', fontsize=16)
    plt.tight_layout(1, rect=(0, 0, 1, 0.95))
    plt.suptitle(u'多项式曲线拟合比较', fontsize=22)
    plt.show()

    y_max = max(max(tss_list), max(essrss_list)) * 1.05
    plt.figure(figsize=(9, 7), facecolor='w')
    t = np.arange(len(tss_list))
    plt.plot(t, tss_list, 'ro-', lw=2, label=u'TSS(Total Sum of Squares)')
    plt.plot(t, ess_list, 'mo-', lw=1, label=u'ESS(Explained Sum of Squares)')
    plt.plot(t, rss_list, 'bo-', lw=1, label=u'RSS(Residual Sum of Squares)')
    plt.plot(t, essrss_list, 'go-', lw=2, label=u'ESS+RSS')
    plt.ylim((0, y_max))
    plt.legend(loc='center right')
    plt.xlabel(u'实验：线性回归/Ridge/LASSO/Elastic Net', fontsize=15)
    plt.ylabel(u'XSS值', fontsize=15)
    plt.title(u'总平方和TSS=？', fontsize=18)
    plt.grid(True)
    plt.show()
    
    t1 = time.time()
    print('Elapsed time is ', t1-t0)
    