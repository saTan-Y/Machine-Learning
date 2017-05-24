#!/usr/bin/python
# -*- coding:utf-8 -*-

from time import time
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    t0 = time()
    df = pd.read_csv('..\\10.Regression\\10.iris.data', header=None)
    df[4] = df[4].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
    x, y = df.values[:, :2], df.values[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

    # model = Pipeline([('sc', StandardScaler()), ('clf', GaussianNB())])
    model = Pipeline([('sc', MinMaxScaler()), ('clf', MultinomialNB())])
    model.fit(x, y.ravel())
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
    xx, yy = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
    grid_test = np.stack((xx.flat, yy.flat), axis=1)
    grid_plot = model.predict(grid_test)
    grid_plot = grid_plot.reshape(xx.shape)

    print(accuracy_score(y_train, model.predict(x_train)), accuracy_score(y_test, model.predict(x_test)))

    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

    plt.figure(facecolor='w')
    plt.pcolormesh(xx, yy, grid_plot, cmap=cm_light)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cm_dark, marker='o', edgecolors='k')
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid()
    plt.title('hh')
    plt.tight_layout()
    plt.show()