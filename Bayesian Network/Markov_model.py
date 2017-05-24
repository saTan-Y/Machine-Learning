#!/usr/bin/python
# -*- coding:utf-8 -*-

from time import time
import numpy as np

if __name__ == '__main__':
    # m, n = 50, 100
    # directions = np.random.rand(m, n, 8)
    # x = np.arange(n)
    # y_d = np.cos(6 * np.pi * x / n)
    # theta = np.empty_like(x, dtype=np.int)
    # theta[y_d > 0.5] = 1
    # theta[~(y_d > 0.5) & (y_d > -0.5)] = 0
    # theta[~(y_d > -0.5)] = 7
    # directions[:, x.astype(np.int), theta] = 10
    # print(directions)
    # print(type(theta))

    a = np.random.rand(3,4,3)
    print(a)
    a[:, np.array([0,1,2,3], dtype=np.int), np.array([0,1,2,0])] = 10
    print(a)