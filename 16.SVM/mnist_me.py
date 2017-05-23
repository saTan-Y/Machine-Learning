#!/usr/bin/python
# -*- coding:utf-8 -*-

from time import time
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
import os
from sklearn.model_selection import  train_test_split

def save_image(x, i):
    x = 255 - x
    a = x.astype(np.uint8)
    out_path = '.\\mnist'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    Image.fromarray(a).save(out_path+'\\%d.png' % (i))

if __name__ == '__main__':
    classifier_type = 'RF'

    t0 = time()
    print('Loading training data started...')
    df = pd.read_csv('16.MNIST.train.csv', header=0, dtype=int)
    x, y = df.values[:, 1:], df.values[:, 0]
    print('Loading training data finished... %f' % (time()-t0))
    print('Figs Num %d Pixel Num %d' % x.shape)
    ig = x.reshape((-1, 28, 28))

    t = time()
    print('Loading test data started...')
    df2 = pd.read_csv('16.MNIST.test.csv', header=0, dtype=int)
    x_t = df2.values
    print('Loading test data finished... %f' % (time() - t))
    print('Figs Num %d Pixel Num %d' % x_t.shape)
    ig_t = x_t.reshape((-1, 28, 28))

    np.random.seed(0)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.4, random_state=1)
    ig_train = x_train.reshape((-1, 28, 28))
    ig_test = x_test.reshape((-1, 28, 28))
    print(ig_train.shape, ig_test.shape)

    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(15, 9), facecolor='w')
    for i in range(16):
        plt.subplot(4, 8, i+1)
        plt.imshow(ig_train[i], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Training Fig %d' % i)
    for i in range(16):
        plt.subplot(4, 8, i+17)
        plt.imshow(ig_test[i], cmap=plt.cm.gray_r, interpolation='nearest')
        save_image(ig_test[i].copy(), i)
        plt.title('Test fig %d' % i)
    plt.tight_layout()
    plt.show()

    if classifier_type == 'SVM':
        model = svm.SVC(C=1000, kernel='rbf', gamma=1e-10)
        t = time()
        print('SVM training started...')
        model.fit(x_train, y_train)
        print('SVM training finished... %f' % (time()-t))

        print('Prediction on training data started...')
        t = time()
        y_pred = model.predict(x_train)
        rate = accuracy_score(y_train, y_pred)
        print('SVM accuracy rate %f, costing %f on training data' % (rate, time()-t))

        print('Prediction on test data started...')
        t = time()
        y_pred2 = model.predict(x_test)
        rate = accuracy_score(y_test, y_pred2)
        print('SVM accuracy rate %f, costing %f on test data' % (rate, time() - t))
    elif classifier_type == 'RF':
        model = RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_split=2, oob_score=True)
        t = time()
        print('RF training started...')
        model.fit(x_train, y_train)
        print('RF training finished... %f' % (time() - t))

        print('Prediction on training data started...')
        t = time()
        y_pred = model.predict(x_train)
        rate = accuracy_score(y_train, y_pred)
        print('SVM accuracy rate %f, costing %f on training data' % (rate, time() - t))

        print('Prediction on test data started...')
        t = time()
        y_pred2 = model.predict(x_test)
        rate = accuracy_score(y_test, y_pred2)
        print('SVM accuracy rate %f, costing %f on test data' % (rate, time() - t))

    error = y_pred2 != y_test
    print(error)
    y_error = y_test[error]
    y_pred_error = y_pred2[error]
    ig_error = ig_test[error]
    plt.figure(figsize=(14, 9), facecolor='w')
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(ig_error[i], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Predicted value %d Real value %d' % (y_pred_error[i], y_error[i]))
    plt.tight_layout()
    plt.show()
    print('Elapsed time is', time() - t0)

