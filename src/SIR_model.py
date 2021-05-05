# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 19:56:23 2020

@author: yiqing
"""

import scipy.integrate
import numpy as np
from scipy.optimize import minimize

class SIR:

    def __init__(self, data, t, N, beta, gamma):

        self.data = data
        self.t = t
        self.N = N
        self.beta = beta
        self.gamma = gamma

    def model(self, data, t, N, beta, gamma):
        """
        :param self:
        :param data:
        :param t:
        :param beta:
        :param gamma:
        :return:
        """

        S, I, R = data
        dS_t = - (beta * S * I) /N
        dI_t = beta * S * I /N - gamma * I
        dR_t = gamma * I

        return ([dS_t, dI_t, dR_t])

    def solve(self, init_value, t, N, beta, gamma):
        """

        :param self:
        :param model:
        :param init_value:
        :param t:
        :param beta:
        :param gamma:
        :return:
        """
        result = scipy.integrate.odeint(self.model, init_value,
                                                 t, args=(N, beta, gamma))
        result = np.array(result)

        return result

    def loss(self, para, real_data):

        size = len(real_data)
        beta, gamma = para
        solution = scipy.integrate.odeint(self.model, self.data, t=np.arange(0, size, 1), args=(self.N, beta, gamma))
        print(beta, beta)
        return np.sqrt(np.mean((solution[:,1] - real_data)**2))

    def train(self, realdata):

        optimal = minimize(
            self.loss,
            [0.001, 0.001],
            args=(realdata),
            method='L-BFGS-B',
            bounds=[(0.00000001, 0.5), (0.00000001, 0.5)]
        )
        beta, gamma = optimal.x

        return beta, gamma