#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from hmmlearn import hmm
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler

def data_clean(state):
    t, y = df['Time'], df['Current']
    idx_time = (t>6.76) & (t<6.91)
    idx_current = (y>9.31) & (y<9.33)
    idx = idx_time & idx_current
    df['Current'][idx] += df['Current'][y>9.34].median() - y[idx].median()

    idx_current = (y > 9.01) & (y < 9.04)
    idx = idx_time & idx_current
    df['Current'][idx] += df['Current'][y > 9.1].median() - y[idx].median()

    if state:
        plt.figure(figsize=(13, 8), facecolor='w')
        plt.plot(t, df['Current'], 'r.-', lw=1)
        plt.ylim(y.min(), y.max())
        plt.xlabel(u'时间')
        plt.ylabel((u'原始电流强度'))
        plt.grid(True, ls=':')
        plt.show()


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    df = pd.read_excel(io='Current.xls', sheetname='Sheet1', header=0)
    # df['Current'] = MinMaxScaler().fit_transform(df['Current'])
    df['Current'] *= 1e6
    data_clean(False)

    n = 3
    x = df['Time'].reshape(-1, 1)
    y = df['Current'].reshape(-1, 1)
    model = hmm.GaussianHMM(n, covariance_type='full', n_iter=10)
    model.fit(y)
    y_pred = model.predict_proba(y)
    y_state = model.predict(y)
    # print(y_pred)
    state = pd.DataFrame(y_pred, index=df.index, columns=np.arange(n))
    data = pd.concat((df, state),axis=1)
    print(data)

    plt.figure(num=1, figsize=(8, 9), facecolor='w')
    plt.subplot(n+1, 1, 1)
    plt.plot(x, y, 'r.-', lw=1)
    plt.ylim(y.min(), y.max())
    plt.xlabel(u'时间')
    plt.ylabel((u'原始电流强度'))
    plt.title('时间到了副驾驶的快乐附件是打开链接发')
    plt.grid(True, ls=':')
    # plt.show()
    for i in range(n):
        plt.subplot(n+1, 1, i+2)
        plt.plot(x, data[i], 'r.')
        plt.xlabel('时间')
        plt.ylabel('组分%d概率' % (i+1))
        plt.title('sdl')
        plt.grid(True, ls=':')
    plt.suptitle('sdfhkosj')
    plt.tight_layout(1)
    # plt.show()

    plt.figure(num=2, figsize=(8, 9), facecolor='w')
    plt.subplot(n + 1, 1, 1)
    plt.plot(x, y, 'r.-', lw=1)
    plt.ylim(y.min(), y.max())
    plt.xlabel(u'时间')
    plt.ylabel((u'原始电流强度'))
    plt.title('时间到了副驾驶的快乐附件是打开链接发')
    plt.grid(True, ls=':')
    # plt.show()
    for i in range(n):
        plt.subplot(n + 1, 1, i + 2)
        plt.plot(y, data[i], 'r.')
        plt.xlabel('电流')
        plt.ylabel('组分%d概率' % (i + 1))
        plt.title('sdl')
        plt.grid(True, ls=':')
    plt.suptitle('sdfhkosj')
    plt.tight_layout(1, rect=(0,0.2,1,0.6))
    # plt.show()

    y_new = np.zeros_like(data['Current'])
    for i in range(n):
        idx = y_state == i
        y_new[idx] = np.median(data['Current'][idx])
    data['new'] = y_new
    data.to_excel('new.xls', index=False)

    plt.figure(num=3, facecolor='w', figsize=(8, 8))
    plt.subplot(211)
    plt.plot(x, y_new, 'r.-', lw=0.2)
    plt.ylim(y_new.min(), y_new.max())
    plt.grid(b=True, ls=':')
    plt.xlabel(u'时间', fontsize=14)
    plt.ylabel(u'电流强度', fontsize=14)
    plt.title(u'整形后的电流数据', fontsize=18)
    plt.subplot(212)
    plt.plot(data['Current'], data['new'], 'r.')
    plt.ylim(data['new'].min(), data['new'].max())
    plt.grid(b=True, ls=':')
    plt.xlabel(u'原始电流强度', fontsize=14)
    plt.ylabel(u'修正后的电流强度', fontsize=14)
    plt.title(u'整形前后的电流关系', fontsize=18)
    plt.tight_layout(pad=1, rect=(0, 0, 1, 1))

    plt.show()