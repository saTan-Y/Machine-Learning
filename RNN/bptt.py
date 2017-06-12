#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from functools import reduce

def element_wise_op(x, operation):
    for i in np.nditer(x, op_flags=['readwrite']):
        i[...] = operation[i]

class RecurrentLayer(object):
    def __init__(self, input_dim, state_dim, activator, learning_rate):
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.activator = activator
        self.learning_rate = learning_rate
        self.time = 0
        self.state_list = np.zeros((state_dim, 1)) #Initialization of state series in time 0
        self.W = np.random.uniform(-1e-3, 1e-3, (state_dim, state_dim))
        self.U = np.random.uniform(-1e-3, 1e-3, (state_dim, input_dim))

    def forward(self, input_vec):
        self.time += 1
        state = (np.dot(self.U, input_vec) + np.dot(self.W, self.state_list[-1]))
        element_wise_op(state, self.activator.forward)
        self.state_list.append(state)

    def bptt(self, sensitivity_array, activator):
        self.calcu_delta(sensitivity_array, activator)
        self.calcu_grad()

    def calcu_delta(self, sensitivity_array, activator):
        self.delta_list = []
        for i in range(self.time):
            self.delta_list.append(np.zeros(self.state_dim, 1))
            self.delta_list.append(sensitivity_array)
            for k in range(self.time -1, 0, -1):
                self.calcu_delta_k(k, activator)

    def calcu_delta_k(self, k, activator):
        state = self.state_list[k+1].copy()
        element_wise_op(self.state_list[k+1], activator.backward)
        self.state_list[k] = np.dot(np.dot(self.state_list[k+1].T, self.W), np.diag(state[:, 0])).T

    def calcu_grad(self):
        self.grad_list = []
        for t in range(self.time + 1):
            self.grad_list.append(np.zeros((self.state_dim, self.state_dim)))
        for t in range(self.time, 0, -1):
            self.calcu_grad_t(t)
        self.grad = reduce(lambda a, b: a+b, self.grad_list, self.grad)

    def calcu_grad_t(self, t):
        grad = np.dot(self.delta_list[t], self.delta_list[t-1].T)
        self.grad_list[t] = grad
    def bpttupdate(self):
        self.W -= self.grad * self.learning_rate

