# -*- coding: utf-8 -*-
"""
SEIR_plusCAQ_model.py
Created on 2020/4/9

@author: yiqing
"""
import scipy.integrate
import numpy as np

class SEIR_plusCAQ:
    def model(data, t, N, alpha, beta, gamma, gamma_a, gamma_q, epsilon,
              eta, theta1, theta2, theta3, rho,  rho1, rho2, rho3,
              phi, omega):
        """
        :param t:
        :param N:
        :param alpha:
        :param beta:
        :param gamma:
        :param gamma_a:
        :param gamma_q:
        :param epsilon:
        :param eta:
        :param theta1:
        :param theta2:
        :param theta3:
        :param rho:
        :param rho1:
        :param rho2:
        :param rho3:
        :param phi:
        :param omega:
        :return:
        """

        S, Sq, E, Eq, I, I_1, I_2, I_3, I_q, R, A= data
        # I = I_1 + I_2 + I_3

        dS_t = (-(1 - rho) * beta * phi * (epsilon * E + I + A) * S -
               rho * beta * (1 - phi) * (epsilon * E + I + A) * S -
               rho * beta * phi * (epsilon * E + I + A) * S)/N + omega * Sq/N
        dE_t = (1 - rho) * beta * phi * (epsilon * E + I + A) * S/N - \
               alpha*(1-eta)*E - alpha * eta * rho1 * E - alpha * eta * rho2 * E - \
               alpha * eta * rho3 * E
        dA_t = alpha * (1 - eta) * E - gamma_a * A
        I = I
        dI_1_t = alpha * eta * rho1 * E - gamma * I_1 * (1-theta1)
        dI_2_t = alpha * eta * rho2 * E - gamma * I_2 * (1-theta2)
        dI_3_t = alpha * eta * rho3 * E - gamma * I_3 * (1-theta3)
        dR_t = gamma_a * A + (1-theta1)*gamma*I_1 + \
               (1-theta2)*gamma*I_2+ (1-theta3)*gamma*I_3 + I_q * gamma_a
        dSq_t = rho * beta * (1 - phi) * (epsilon * E + I + A) * S/N - omega * Sq/N
        dEq_t = rho * beta * phi * (epsilon * E + I + A) * S/N - alpha * Eq
        dI_q_t = alpha * Eq + theta1 * gamma * I_1 + theta2 * gamma * I_2 \
                 + theta3 * gamma * I_3 - gamma_q * I_q

        return ([dS_t, dSq_t, dE_t, dEq_t, I, dI_1_t, dI_2_t, dI_3_t, dI_q_t, dR_t,dA_t])

    def solve(init_value, t, N, alpha, beta, gamma, gamma_a, gamma_q, epsilon,
              eta, theta1, theta2, theta3, rho,  rho1, rho2, rho3,
              phi, omega):
        """
        :param t:
        :param N:
        :param alpha:
        :param beta:
        :param gamma:
        :param gamma_a:
        :param gamma_q:
        :param epsilon:
        :param eta:
        :param theta1:
        :param theta2:
        :param theta3:
        :param rho:
        :param rho1:
        :param rho2:
        :param rho3:
        :param phi:
        :param omega:
        :return:
        """

        result = scipy.integrate.odeint(SEIR_plusCAQ.model, init_value,
                                        t, args=(N, alpha, beta, gamma, gamma_a, gamma_q, epsilon,
                                                 eta, theta1, theta2, theta3, rho,  rho1, rho2, rho3,
                                                 phi, omega))

        result = np.array(result)

        return result













