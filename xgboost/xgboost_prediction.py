#!/usr/bin/python
# -*- coding:utf-8 -*-

import time
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    t0 = time.time()
    df = pd.read_csv('..\\10.Regression\\10.iris.data', header=None)
    df[4] = df[4].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}).astype(int)
    x, y = df.values[:, :4], df.values[:, -1]
    # print(x, y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=50, random_state=1)

    train_data = xgb.DMatrix(x_train, label=y_train)
    test_data = xgb.DMatrix(x_test, label=y_test)
    params = {'max_depth': 2, 'eta': 0.3, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 3}
    watch_list = [(test_data, 'eval'), (train_data, 'train')]
    bst = xgb.train(params, train_data, num_boost_round=5, evals=watch_list)

    y_pred = bst.predict(test_data)
    acc_rate = accuracy_score(y_test.ravel(), y_pred)
    print(acc_rate)
    print('Elapsed time is', time.time()-t0)
