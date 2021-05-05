# -*- coding: utf-8 -*-
"""
SEIR_plusCQD_model.py
Created on 2020/4/10

@author: yiqing
"""
import scipy.integrate
import numpy as np


class SEIR_plusCQD:

    def model(data, t, N, alpha, beta, gamma, delta, lamda, kappa):
        """
        :param data: [S, P, E, I, Q, R ,D]
        :param t:
        :param N:
        :param alpha:
        :param beta:
        :param gamma:
        :param delta:
        :param lumda:
        :param kappa:
        :return:
        """
        S, P, E, I, Q, R, D = data
        dS_t = - (beta * S * I) / N - alpha * S
        dE_t = beta * S * I / N - gamma * E
        dI_t = gamma * E - delta * I
        dQ_t = delta * I - lamda * Q - kappa * Q
        dR_t = lamda * Q
        dD_t = kappa * Q
        dP_t = alpha * S
        # print([dS_t, dE_t, dI_t, dQ_t, dR_t, dD_t, dP_t])
        return [dS_t, dP_t, dE_t, dI_t, dQ_t, dR_t, dD_t]

    def solve(init_value, t, N, alpha, beta, gamma, delta, lamda, kappa):
        """
        :param t:
        :param N:
        :param alpha:
        :param beta:
        :param gamma:
        :param delta:
        :param lumda:
        :param kappa:
        :return:
        """
        result = scipy.integrate.odeint(SEIR_plusCQD.model, init_value, t, args=(N, alpha, beta,
                                                                                 gamma, delta, lamda, kappa))
        result = np.array(result)
        return result

