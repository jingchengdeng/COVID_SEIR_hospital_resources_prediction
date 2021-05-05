# -*- coding: utf-8 -*-
"""
SEIR_model.py
Created on 2020/4/8

@author: yiqing
"""

import scipy.integrate
import numpy as np

class SEIR:
    def model(data, t, N, beta, gamma, sigma):
        """
        :param data:
        :param t:
        :param N:
        :param beta:
        :param gamma:
        :param sigma:
        :return:
        """

        S, E, I, R = data

        dS_t = - (beta * S * I)/N
        dE_t = beta * S * I/N - sigma * E
        dI_t = sigma * E - gamma * I
        dR_t = gamma * I

        return [dS_t, dE_t, dI_t, dR_t]

    def solve(init_value, t, N, beta, gamma, sigma):
        """
        :param model: SEIR model
        :param init_value: S,E,I,R = population - E0, E0, 0, 0
        :param t:
        :param beta: The parameter controlling how often a susceptible-infected contact results in a new infection.
        :param gamma: The rate an infected recovers and moves into the resistant phase.
        :param sigma: The rate at which an exposed person becomes infective.
        :return:
        """

        result = scipy.integrate.odeint(SEIR.model, init_value, t, args=(N, beta, gamma, sigma))
        result = np.array(result)
        return result