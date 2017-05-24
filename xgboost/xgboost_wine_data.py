#!/usr/bin/python
# -*- coding:utf-8 -*-

import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    t0 = time.time()
    df = pd.read_csv('14.wine.data', dtype=float, header=None)
    df[0] = df[0].map({1.0: 0, 2.0: 1, 3.0: 2}).astype(int)
    x, y = df.values[:, 1:], df.values[:, 0]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=1)

    lr = LogisticRegression(penalty='l2')
    lr.fit(x_train, y_train.ravel())
    y_pred1 = lr.predict(x_test)
    rate1 = accuracy_score(y_test, y_pred1)

    train_data = xgb.DMatrix(x_train, label=y_train)
    test_data = xgb.DMatrix(x_test, label=y_test)
    params = {'max_depth': 2, 'eta':0.2, 'silent': 0, 'objective':'multi:softmax', 'num_class': 3}
    watch_list = [(test_data, 'eval'), (train_data, 'train')]
    bst = xgb.train(params, train_data, num_boost_round=10, evals=watch_list)
    y_pred2 = bst.predict(test_data)
    rate2 = accuracy_score(y_test, y_pred2)
    print('lr rate', rate1)
    print('xgboost rate', rate2)
    print('Elapsed time is', time.time()-t0)
