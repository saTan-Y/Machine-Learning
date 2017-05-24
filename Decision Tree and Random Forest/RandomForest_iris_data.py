# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 15:47:05 2017

@author: saTan-Y
"""

import numpy as np
import pandas as pd
import time
import pydotplus
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#from sklearn import tree
from sklearn.preprocessing import LabelEncoder
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.preprocessing import StandardScaler
#from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
    t0 = time.time()
    
    iris_feature_E = 'sepal length', 'sepal width', 'petal length', 'petal width'
    iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'
    iris_class = 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'

    path = '10.iris.data'
    df = pd.read_csv(path, header=None)
    x, y = df.values[:,:3], df.values[:,-1]
    xx = x[:,:2]
    yy = LabelEncoder().fit_transform(y)
    
    x_train, x_test, y_train, y_test = train_test_split(xx, yy, train_size=0.75, random_state=1)
    model = RandomForestClassifier(n_estimators=200, criterion='entropy', max_depth=5)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
#    with open('iris.dot', 'w') as f:
#        tree.export_graphviz(model, out_file=f)
#        
#    data = tree.export_graphviz(model, out_file=None, feature_names=iris_feature_E, class_names=iris_class, filled=True, rounded=True, special_characters=True)
#    graph = pydotplus.graph_from_dot_data(data)
#    graph.write_pdf('iris.pdf')
#    f = open('iris.png', 'wb')
#    f.write(graph.create_png())
#    f.close()
    
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    M = 500
    x1_min, x1_max = xx[:,0].min(), xx[:,0].max()
    x2_min, x2_max = xx[:,1].min(), xx[:,1].max()
    ax0 = np.linspace(x1_min-0.3, x1_max+0.3, M)
    ax1 = np.linspace(x2_min-0.3, x2_max+0.3, M)
    x1, x2 = np.meshgrid(ax0, ax1)
    xx_2 = np.stack((x1.flat, x2.flat), axis=1)
    y_pred2 = model.predict(xx_2)
    y_pred2 = y_pred2.reshape(x1.shape)
    
    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    plt.pcolormesh(x1, x2, y_pred2, cmap = cm_light)
    plt.scatter(x_train[:,0], x_train[:,1], c=y_train.ravel(), cmap=cm_dark, edgecolors='k', s=50)
    plt.scatter(x_test[:,0], x_test[:,1], c=y_test.ravel(), cmap=cm_dark, edgecolors='k', marker='*', s=50)
    plt.xlabel(iris_feature[0])
    plt.ylabel(iris_feature[1])
    plt.xlim(x1_min-0.3, x1_max+0.3)
    plt.ylim(x2_min-0.3, x2_max+0.3)
    plt.tight_layout()
    plt.show()
    corr_rate = 100*np.mean(y_pred == y_test)
    print('Correction rate is ', corr_rate)
    
    err = []
    for d in range(1,15):
        model2 = RandomForestClassifier(n_estimators=200, criterion='entropy', max_depth=d)
        model2.fit(x_train, y_train)
        y_pred3 = model2.predict(x_test)
        err_rate = 100*(1-np.mean(y_test == y_pred3))
        err.append(err_rate)
    plt.figure()
    plt.plot(range(1, 15), err, lw=2, alpha=0.7)
    plt.xlabel(u'回合数')
    plt.ylabel(u'错误率')
    plt.title('sjldkfj')
    plt.show()
    
    estimator_num = np.linspace(10,310,20).astype('int')
    model3 = GridSearchCV(RandomForestClassifier(), param_grid={'n_estimators':estimator_num, 'criterion':['entropy'], 'max_depth':[5]}, cv=5)
    model3.fit(x_train, y_train)
    yyy = model3.predict(x_test)
    print('Best parameter is', model3.best_params_)
    print('rate of gdcv is ', 100*np.mean(yyy == y_test))
    
    t1 = time.time()
    print('Elapsed time is ', t1-t0)