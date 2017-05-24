# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 14:53:10 2017

@author: saTan-Y
"""

import numpy as np
import matplotlib as mpl
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    t0 = time.time()
    iris_feature = [u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度']
    
    df = pd.read_csv('10.iris.data')
    x, y = df.values[:,:-1], df.values[:,-1]
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    mpl.rcParams['font.sans-serif'] = [u'SimHei']  # 黑体 FangSong/KaiTi
    mpl.rcParams['axes.unicode_minus'] = False
                
    feature_list = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    plt.figure()
    for i, d in enumerate(feature_list):
        xx = x[:,d]
        model = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=3)
        model = model.fit(xx, y)
        y_pred = model.predict(xx)
        cor_num = np.count_nonzero(y_pred == y)
        cor_rate = 100*cor_num/len(y)
        print('features are %s and %s' % (iris_feature[d[0]], iris_feature[d[1]]))
        print('Correction number is ', cor_num)
        print('correction rate is ', cor_rate)
        
        M = 500
        x1_min, x1_max = x[:,d[0]].min(), x[:,d[0]].max()
        x2_min, x2_max = x[:,d[1]].min(), x[:,d[1]].max()
        ax1 = np.linspace(x1_min, x1_max)
        ax2 = np.linspace(x2_min, x2_max)
        x1, x2 = np.meshgrid(ax1, ax2)
        xx = np.stack((x1.flat, x2.flat), axis=1)
        yy = model.predict(xx)
        yy = yy.reshape(x1.shape)
        
        cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
        cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
        plt.subplot(3, 2, i+1)
        plt.pcolormesh(x1, x2, yy, cmap=cm_light)
        plt.scatter(x[:, d[0]], x[:,d[1]], c=y_pred.ravel(), cmap=cm_dark, edgecolors='k')
        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)
        plt.xlabel(iris_feature[d[0]], fontsize=14)
        plt.ylabel(iris_feature[d[1]], fontsize=14)
        plt.tight_layout()
        plt.show()
    
    t1 = time.time()
    print('Elapsed time is ', t1-t0)