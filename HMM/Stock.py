#!/usr/bin/python
# -*- coding:utf-8 -*-

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from hmmlearn import hmm
import warnings

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    df = pd.read_table('26.SH600000.txt', sep='\t', encoding='gb2312', header=1)
    df.columns = ['date', 'begin', 'highest', 'lowest', 'end', 'volume', 'amount']
    df['amp_price'] = df['highest'] - df['lowest']
    diff_price = np.diff(df['end'])

    samples = np.column_stack((diff_price, df['volume'][1:], df['amount'][1:], df['amp_price'][1:]))
    n = 5
    model = hmm.GaussianHMM(n_components=5, covariance_type='full')
    model.fit(samples)
    y = model.predict_proba(samples)
    print(y)

    t = range(len(diff_price))
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 8), facecolor='w')
    plt.subplot(421)
    plt.plot(t, diff_price, 'r-')
    plt.title('u涨跌幅')
    plt.subplot(422)
    plt.plot(t, df['volume'][1:], 'g-')
    plt.title('u成交量')

    colors = mpl.cm.terrain(np.arange(0, 0.8, n))
    plt.subplot(423)
    for i, item in enumerate(colors):
        plt.plot(t, y[:, i], c=item)
    plt.title('u所有组分')
    for i in range(4, 9):
        plt.subplot(4, 2, i)
        plt.plot(t, y[:, i-4])
        plt.title('u组分%d' % (i-3))
    plt.tight_layout(1)
    plt.show()