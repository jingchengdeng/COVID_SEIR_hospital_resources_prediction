# -*- coding: utf-8 -*-
"""
SEIR_plusBG_model.py
Created on 2020/4/11

@author: yiqing
"""
import scipy.integrate
import numpy as np

class SEIR_plusBG:

    def model(data, t, N, beta_1, beta_2, gamma_1, gamma_2, phi,
              omega, tau, alpha, rho, kappa, lamda_1, lamda_2):

        S, E, Eq, Sq, Ih, Im, Iq, D, R = data

        dS_t = - beta_1 * Im * S * gamma_1 / N - beta_2 * E * S * gamma_2 / N - phi * S + omega * Sq + R * tau
        dE_t = (beta_1 * Im * S * gamma_1 / N + beta_2 * E * S * gamma_2 / N) * (1 - phi) - alpha * E
        dEq_t = (beta_1 * Im * S * gamma_1 / N + beta_2 * E * S * gamma_2 / N) * phi - alpha * Eq
        dSq_t = phi * S - omega * Sq
        dIh_t = alpha * E * rho + alpha * Eq * rho - kappa * Ih - lamda_1 * Ih
        dIm_t = alpha * E * (1 - rho) - lamda_2 * Im
        dIq_t = alpha * Eq * (1 - rho) - lamda_2 * Iq
        dD_t = kappa * Ih
        dR_t = lamda_1 * Ih + lamda_2 * Iq + lamda_2 * Im - R * tau

        return [dS_t, dE_t, dEq_t, dSq_t, dIh_t, dIm_t, dIq_t, dD_t, dR_t]

    def solve(init_value, t, N, beta_1, beta_2, gamma_1, gamma_2, phi,
              omega, tau, alpha, rho, kappa, lamda_1, lamda_2):

        result = scipy.integrate.odeint(SEIR_plusBG.model, init_value, t, args=(N, beta_1, beta_2, gamma_1,
                                                                                gamma_2, phi, omega, tau,
                                                                                alpha, rho, kappa, lamda_1,
                                                                                lamda_2))
        result = np.array(result)
        return result
