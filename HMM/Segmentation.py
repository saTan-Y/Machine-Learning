#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import random
import time

infinite = -(2**31)

def log_normalize(a):
    s = sum(a)
    s = np.log(s)
    for i in range(len(a)):
        if a[i] == 0:
            a[i] = infinite
        else:
            a[i] = np.log(a[i]) - s

def list_write(f, v):
    for a in v:
        f.write(str(a))
        f.write(' ')
    f.write('\n')

def load_train():
    f = open('pi.txt','r')
    for line in f:
        pi = list(map(float, line.split(' ')[:-1]))
    f.close()

    f = open('A.txt','r')
    A = [[] for x in range(4)]
    i = 0
    for line in f:
        A[i] = list(map(float, line.split(' ')[:-1]))
        i += 1
    f.close()

    f = open('B.txt', 'r')
    B = [[] for x in range(4)]
    i = 0
    for line in f:
        B[i] = list(map(float, line.split(' ')[:-1]))
        i += 1
    f.close()
    return pi, A, B

def viterbi(pi, A, B, o):
    T = len(o)
    delta = [[0] * 4 for x in range(T)]
    record = [[0] * 4 for x in range(T)]
    for i in range(4):
        delta[0][i] = pi[i] + B[i][ord(o[0])]
    for t in range(1, T):
        for i in range(4):
            temp = [delta[t-1][j] + A[j][i] for j in range(4)]
            temp2 = np.array(temp)
            max_delta, jj = temp2.max(), temp2.argmax()
            delta[t][i] = max_delta + B[i][ord(o[t])]
            record[t][i] = jj
    decoding = [-1] * T
    # q = 0
    temp3 = np.array(delta[T-1])
    q = decoding[T-1] = temp3.argmax()
    for t in range(T-2, -1, -1):
        q = record[t+1][q]
        decoding[t] = q
    return decoding

def segment(txt, decoding):
    N = len(txt)
    i = 0
    while i < N:
        if decoding[i] == 0 or decoding[i] == 1:
            j = i + 1
            while j < N:
                if decoding[j] == 2:
                    break
                j += 1
            print(txt[i:j+1],'|',end='')
            i = j + 1
        elif decoding[i] == 2 or decoding[i] == 3:
            print(txt[i],'|',end='')
            i += 1
        else:
            print('Error!')
            i += 1

if __name__ == "__main__":
    t0 = time.time()
    pi, A, B = load_train()
    # print(type(pi[0]))
    f = open('26.novel.txt','r',encoding='utf-8')
    data = f.read()[10:]
    f.close()
    decoding = viterbi(pi, A, B, data)
    # print(decoding)
    segment(data, decoding)
    print('\nElapsed time is', time.time() - t0)