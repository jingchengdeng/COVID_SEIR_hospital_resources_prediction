# -*- coding: utf-8 -*-
"""
SEIR_plus_model.py
Created on 2020/4/11

@author: yiqing
"""
import scipy.integrate
import numpy as np
from scipy.optimize import minimize


class SEIR_plus:

    def __init__(self, data, t, N, beta1, beta2, sigma, alpha, rho, rho_icu, rho_v,
                 lamda1, lamda2, lamda_icu, lamda_v, kappa):
        """
        :param data: [S, E, Im, Ih, Iicu, Iv, D, R]
        :param t:
        :param N:
        :param beta1:
        :param beta2:
        :param sigma:
        :param alpha:
        :param rho:    hospitalized %
        :param rho_icu: icu %
        :param rho_v:  ventilator %
        :param lamda1: Im reverse of recover time
        :param lamda2: Ih reverse of recover time
        :param lamda_icu:
        :param lamda_v:
        :param kappa:
        """

        self.data = data
        self.t = t
        self.N = N
        self.beta1 = beta1
        self.beta2 = beta2
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.rho_icu = rho_icu
        self.rho_v = rho_v
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.lamda_icu = lamda_icu
        self.lamda_v = lamda_v
        self.kappa = kappa

    def model(self, data, t, N, beta1, beta2, sigma, alpha, rho, rho_icu, rho_v,
              lamda1, lamda2, lamda_icu, lamda_v, kappa):
        """
        :param t:
        :param N:
        :param beta1:
        :param beta2:
        :param sigma:
        :param alpha:
        :param rho:
        :param rho_icu:
        :param rho_v:
        :param lamda1:
        :param lamda2:
        :param lamda_icu:
        :param lamda_v:
        :param kappa:
        :return: array of derivative
        """
        S, E, Im, Ih, Iicu, Iv, D, R = data
        dS_t = - beta1 * Im * S / N - beta2 * E * S / N + sigma * R
        dE_t = beta1 * Im * S / N + beta2 * E * S / N - alpha * E
        dIm_t = alpha * E * (1 - rho) - lamda1 * Im
        dIh_t = alpha * E * rho - lamda2 * Ih - kappa * Ih
        dIicu_t = Ih * rho_icu - lamda_icu * Iicu
        dIv_t = Ih * rho_v - lamda_v * Iv
        # print("day", t)
        # print("dIh", dIh_t)
        # print("dvent", dIv_t)
        dD_t = kappa * Ih
        dR_t = lamda1 * Im + lamda2 * Ih - sigma * R

        return [dS_t, dE_t, dIm_t, dIh_t, dIicu_t, dIv_t, dD_t, dR_t]

    def solve(self):
        """
        :return: array of predict result
        """

        result = scipy.integrate.odeint(self.model, self.data, self.t, args=(self.N, self.beta1, self.beta2, self.sigma,
                                                                             self.alpha, self.rho, self.rho_icu,
                                                                             self.rho_v,
                                                                             self.lamda1, self.lamda2, self.lamda_icu,
                                                                             self.lamda_v, self.kappa))

        result = np.array(result)
        return result

    def loss(self, para, real_data):
        """
        RMSE between actual confirmed and the estimated
        :param para:
        :param real_data:
        :param real_I:
        :return: RMSE
        """
        size = len(real_data)
        beta1, beta2, alpha, lamda1, rho, kappa, sigma = para
        solution = scipy.integrate.odeint(self.model, self.data, t=np.arange(0, size, 1), args=(self.N, beta1, beta2,
                                                                                                sigma,
                                                                                                alpha, rho,
                                                                                                self.rho_icu,
                                                                                                self.rho_v,
                                                                                                lamda1, self.lamda2,
                                                                                                self.lamda_icu,
                                                                                                self.lamda_v, kappa))

        l1 = np.sqrt(np.mean((solution[:, 3] - real_data) ** 2))
        # l2 = np.sqrt(np.mean((solution[:, 3] + solution[:, 2] - real_I) ** 2))

        # return 0.2 * l1 + (1 - 0.8) * l2

        return np.sqrt(np.mean((solution[:, 3] - real_data) ** 2))

    def train(self, realdata):
        """
        calculate optimal parameters
        :param realdata:  array of validate data
        :param real_I:  array of validate data
        :return: optimal parameters
        """
        optimal = minimize(
            self.loss,
            [0.000000000001, 0.000000000001, 0.001, 0.001, 0.025, 0.025, 0.001],
            args=(realdata),
            method='L-BFGS-B',  # Nelder-Mead L-BFGS-B
            bounds=[(0.000000000001, 0.1), (0.000000000001, 0.1), (0.00001, 1),
                    (0.00001, 1), (0.00001, 0.5), (0.00001, 1), (0.00001, 1)]
        )
        beta1, beta2, alpha, lamda1, rho, kappa, sigma = optimal.x
        return beta1, beta2, alpha, lamda1, rho, kappa, sigma
