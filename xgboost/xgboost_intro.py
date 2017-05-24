#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import xgboost as xgb
import time

if __name__ == '__main__':
    t0 = time.time()
    train_data = xgb.DMatrix('14.agaricus_train.txt')
    test_data = xgb.DMatrix('14.agaricus_test.txt')
    print(train_data)
    print(type(train_data))

    parameters = {'max_depth': 2, 'eta': 0.3, 'silent': 1, 'objective':'binary:logistic'}
    watch_list = [(test_data, 'eval'), (train_data, 'train')]
    num_round = 5
    bst = xgb.train(parameters, train_data, num_round, watch_list)

    y_pred = bst.predict(test_data)
    y = test_data.get_label()
    error_rate = sum(y!=(y_pred>0.5)) / float(len(y_pred))
    print(y_pred)
    print('error rate is:', error_rate)