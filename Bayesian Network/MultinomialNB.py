#!/usr/bin/python
# -*- coding:utf-8 -*-

from time import time
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    t0 = time()
    np.random.seed(0)
    M, N = 200, 10
    x = np.random.randint(2, size=(M, N))
    x = np.array(list(set([tuple(t) for t in x])))
    M = len(x)
    y = [0, 1, 2] * int(M/2)
    y = np.array(y[:M])
    print(x.shape)

    # model = MultinomialNB(alpha=1)
    model = GaussianNB()
    model.fit(x, y)
    y_pred = model.predict(x)
    print(accuracy_score(y, y_pred))
    print(model.score(x, y))
