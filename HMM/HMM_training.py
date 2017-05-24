#!/usr/bin/python
# -*- coding:utf-8 -*-

import time
import numpy as np
import math

infi = float(-2**31)

def log_norm(a):
    # aa = np.array(a, dtype=float)
    s = sum(a)
    if s == 0:
        print('Error! The value cannot be zero')
        return
    ss = np.log(s)
    # a = np.array(a)
    # b = np.zeros((len(a), 1), dtype=float)
    for i in range(len(a)):
        if a[i] == 0:
            a[i] = infi
        else:
            a[i] = np.log(a[i]) - ss
    # a[a == 0] = infi
    # a[a != 0] = np.log(a[a != 0]) - ss
    return a

def log_sum(x):
    a = np.array(x)
    m = np.max(a)
    s = np.sum(np.exp(a - m))
    s2 = m + np.log(s)
    return s2

def coe_alpha(pi, A, B, o, num_I):
    T = len(o)
    alpha = np.zeros((T, num_I))
    B = np.array(B)
    pi = np.array(pi)
    A = np.array(A)
    for i in range(num_I):
        alpha[0][i] = pi[i] + B[i][ord(o[0])]
    for t in range(1, T):
        for n in range(num_I):
            temp = alpha[t-1]*A.T[n]
            alpha[t][n] = log_sum(temp) + B[n][ord(o[t])]
    return alpha

def coe_beta(pi, A, B, o, num_I):
    T = len(o)
    beta = np.zeros((T, num_I))
    B = np.array(B)
    pi = np.array(pi)
    A = np.array(A)
    for i in range(num_I):
        beta[T-1][i] = 1
    for t in range(T-2, -1, -1):
        for n in range(num_I):
            for j in range(num_I):
                temp_beta[j] = beta[t+1][j] + A[n][j] + B[j][ord(o[t+1])]
            beta[t][n] = log_sum(temp_beta)
    return beta

def coe_gamma(alpha, beta):
    T = len(alpha)
    gamma = np.zeros((T, len(alpha[0])))
    temp = np.zeros((T, len(alpha[0])))
    for t in range(T):
        for i in range(len(alpha[0])):
            temp[t][i] = alpha[t][i] + beta[t][i]
        s = log_sum(temp[t])
        gamma[t] = temp[t][i] - s
    return gamma

def coe_ksi(alpha, beta, A, B, o):
    B = np.array(B)
    A = np.array(A)
    T = len(o)
    num_I = len(A[0])
    ksi = np.zeros((T, num_I, num_I))
    temp = np.zeros((T, num_I, num_I))
    for t in range(T_1):
        for i in range(num_I):
            for j in range(num_I):
                temp[t][i][j] = alpha[t][i] + A[i][j] + beta[t+1][j] + B[j][ord(o[t+1])]
        s = log_sum(temp[t])
        for i in range(num_I):
            for j in range(num_I):
                ksi[t][i][j] = temp[t][i][j] - s

def bw(alpha, beta, gamma, ksi, o):
    T = len(o)
    N = len(alpha[0])
    pi = np.zeros((T, 1))
    A = np.zeros((N, N))
    B = np.zeros((N, 65536))
    temp1 = np.zeros((1, T))
    temp2 = np.zeros((1, T))
    for i in range(N):
        pi[i] = gamma[i]
    for i in range(N):
        for j in range(N):
            A[i][j] = log_sum(ksi[:,i,j]) - log_sum(gamma[:,i])
    for i in range(N):
        for k in range(65536):
            valid = 0
            for t in range(T):
                if ord(o[t]) == k:
                    temp1[valid] = gamma[i][t]
                    valid += 1
                temp2[t] = gamma[i][t]
            if valid == 0:
                B[i][k] = infi
            else:
                B[i][k] = log_sum(temp1[:valid]) - log_sum(temp2)
    return pi, A, B

def baum_welch(pi, A, B):
    f = file('.\\1.txt')
    sentences = f.read()[3:].decode('utf-8')
    f.close()
    T = len(sentences)
    for time in range(5):
        alpha = coe_alpha(pi, A, B, sentences, 4)
        beta = coe_beta(pi, A, B, sentences, 4)
        gamma = coe_gamma(alpha, beta)
        ksi = coe_ksi(alpha, beta, A, B, sentences)
        pi, A, B = bw(alpha, beta, gamma, ksi, sentences)
    return pi, A, B

def mle():
    # pi = np.zeros((4, 1), dtype=float)
    # A = np.zeros((4, 4), dtype=float)
    # B = np.zeros((4, 65536), dtype=float)
    pi = [0] * 4
    A = [[0] * 4 for i in range(4)]
    B = [[0] * 65536 for i in range(4)]
    f = open('26.pku_training.utf8', 'r', encoding='utf-8')
    data = f.read()[114:]
    f.close()
    tokens = data.split('  ')
    # f = open('26.Englishword.train', 'r', encoding='utf-8')
    # data = f.read()
    # f.close()
    # tokens.extend(data.split(' '))

    last_q = 2
    last_pro = 0
    print('进度：')
    for k, token in enumerate(tokens):
        pro = float(k) / float(len(tokens))
        if pro - last_pro > 0.1:
            print('%.2f%%' % (100*pro))
            last_pro = pro
        token = token.strip()
        n = len(token)
        if n == 0:
            continue
        if n == 1:
            pi[3] += 1
            A[last_q][3] += 1
            B[3][ord(token[0])] += 1
            last_q = 3
            continue
        pi[0] += 1
        pi[2] += 1
        pi[1] += (n-2)
        A[last_q][0] += 1
        last_q = 2
        if n == 2:
            A[0][2] += 1
        else:
            A[0][1] += 1
            A[1][2] += 1
            A[1][1] += (n-3)
        B[0][ord(token[0])] += 1
        B[2][ord(token[n-1])] += 1
        for i in range(1, n-1):
            B[1][ord(token[i])] += 1

    pi = log_norm(pi)
    for i in range(4):
        A[i] = log_norm(A[i])
        B[i] = log_norm(B[i])
    return pi, A, B

def save_list(file, v):
    for item in v:
        file.write(str(item))
        file.write(' ')
    file.write('\n')

def save_para(pi, A, B):
    file_pi = open('.\\pi.txt','w')
    save_list(file_pi, pi)
    file_pi.close()
    file_A = open('.\\A.txt','w')
    for a in A:
        save_list(file_A, a)
    file_A.close()
    file_B = open('.\\B.txt','w')
    for b in B:
        save_list(file_B, b)
    file_B.close()

if __name__ == '__main__':
    pi, A, B = mle()
    print(pi, A)
    # print(pi.size)
    save_para(pi, A, B)
    # temp = log_norm(np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],dtype=float)[0])
    # A = np.zeros((4,4))
    # A[0] = temp
    # print(A)



