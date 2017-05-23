#!/usr/bin/python
# -*- coding:utf-8 -*-

from time import time
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from PIL import Image
import os

def save_image(x, i):
    x *= 15.9375
    x = 255 - x
    a = x.astype(np.uint8)
    out_path = '.\\HandWrittenPic'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    Image.fromarray(a).save(out_path + ('\\%d.png' % i))

if __name__ == '__main__':
    t0 = time()
    print('Loading train data started...')
    df = pd.read_csv('16.optdigits.tra', dtype=float, header=None)
    x, y = df.values[:, :-1], df.values[:, -1]
    ig = x.reshape((-1, 8, 8))
    y = y.ravel().astype(int)
    print('Loading train data finished')

    print('Loading test data started...')
    df2 = pd.read_csv('16.optdigits.tes',dtype=float, header=None)
    x2, y2 = df2.values[:, :-1], df2.values[:, -1]
    ig_t = x2.reshape((-1, 8, 8))
    y_t = y2.ravel().astype(int)
    print('Loading test data finished')

    #Train figs and test figs
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(15, 9), facecolor='w')
    for i in range(16):
        plt.subplot(4, 8, i+1)
        plt.imshow(ig[i], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Train fig %d' % i)
    for i in range(16):
        plt.subplot(4, 8, i+17)
        plt.imshow(ig_t[i], cmap=plt.cm.gray_r, interpolation='nearest')
        save_image(ig_t[i].copy(), i)
        plt.title('Test fig %d' % i)
    plt.tight_layout()
    plt.show()

    model = svm.SVC(C=10, kernel='rbf', gamma=0.001)
    print('Training started...')
    t = time()
    model.fit(x, y)
    print('Training finished')
    print('Training costs', time()-t)

    y_pred = model.predict(x2)
    print('accuracy rate', accuracy_score(y2, y_pred))

    error = y_pred != y2
    print(error)
    y_error = y2[error]
    y_pred_error = y_pred[error]
    ig_error = ig_t[error]
    plt.figure(figsize=(14, 9), facecolor='w')
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(ig_error[i], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Predicted value %d Real value %d' % (y_pred_error[i], y_error[i]))
    plt.tight_layout()
    plt.show()
    print('Elapsed time is', time()-t0)




