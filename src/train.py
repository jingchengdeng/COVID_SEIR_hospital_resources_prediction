# -*- coding: utf-8 -*-
"""
train.py
Created on 2020/5/4

@author: yiqing
"""
from SEIR_plus_model import SEIR_plus
import numpy
import pandas as pd
from matplotlib.pyplot import *
import matplotlib.pyplot as plt

class Train:

    def __init__(self, predict_len, real_len, realdata, real_icu, real_vent, seir, N, beta1, beta2, sigma, alpha, rho,
                 rho_icu, rho_v,
                 lamda1, lamda2, lamda_icu, lamda_v, kappa, hospital_market_share):
        """
        :param predict_len:
        :param real_len:
        :param realdata:
        :param real_icu:
        :param real_vent:
        :param seir:
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
        :param hospital_market_share:
        """
        self.real_vent = real_vent
        self.real_icu = real_icu
        self.hospital_market_share = hospital_market_share
        self.realdata = realdata
        self.kappa = kappa
        self.lamda_v = lamda_v
        self.lamda_icu = lamda_icu
        self.lamda2 = lamda2
        self.lamda1 = lamda1
        self.rho_v = rho_v
        self.rho_icu = rho_icu
        self.rho = rho
        self.alpha = alpha
        self.sigma = sigma
        self.beta2 = beta2
        self.beta1 = beta1
        self.N = N
        self.seir = seir
        self.predict_len = predict_len
        self.real_len = real_len

    def train(self):
        """
        :return:
        """
        t = numpy.linspace(0, self.predict_len, self.predict_len)
        SEIR_MHD = SEIR_plus(self.seir, t, self.N, self.beta1, self.beta2, self.sigma, self.alpha, self.rho,
                             self.rho_icu, self.rho_v,
                             self.lamda1, self.lamda2, self.lamda_icu, self.lamda_v, self.kappa)

        beta1, beta2, alpha, lamda1, rho, kappa, sigma = SEIR_MHD.train(self.realdata / self.hospital_market_share)

        SEIR_MHD = SEIR_plus(self.seir, t, self.N, beta1, beta2, sigma, alpha, rho, self.rho_icu, self.rho_v,
                             lamda1, self.lamda2, self.lamda_icu, self.lamda_v, kappa)

        solution = SEIR_MHD.solve()
        return solution

    def cal_PI(self, solution):
        """
        :param solution:
        :return:
        """
        size = len(self.realdata)
        RMSE_Ih = numpy.sqrt(numpy.mean((solution[:, 3][:size] * self.hospital_market_share - self.realdata) ** 2))
        PI_Ih = 1.96 * RMSE_Ih
        RMSE_Iicu = numpy.sqrt(numpy.mean((solution[:, 4][:size] * self.hospital_market_share - self.real_icu) ** 2))
        PI_Iicu = 1.96 * RMSE_Iicu
        RMSE_Iv = numpy.sqrt(numpy.mean((solution[:, 5][:size] * self.hospital_market_share - self.real_vent) ** 2))
        PI_Iv = 1.96 * RMSE_Iv

        return PI_Ih, PI_Iicu, PI_Iv

    def output(self, date, date_real, hospital_market_share, realdata, real_icu, real_vent):
        """
        :param date:
        """
        solution = self.train()
        PI_Ih, PI_Iicu, PI_Iv = self.cal_PI(solution)
        output_data = {'Date': date,
                       'Number of Current All Beds In Use': self.realdata,
                       'Number of All Beds Forecast': numpy.ceil(solution[:, 3] * self.hospital_market_share),
                       'Number of Current Icu Beds In Use': self.real_icu,
                       'Number of ICU Beds Forecast': numpy.ceil(solution[:, 4] * self.hospital_market_share),
                       'Number of Current Ventilators In Use': self.real_vent,
                       'Number of Ventilators Forecast': numpy.ceil(solution[:, 5] * self.hospital_market_share)}

        output_df = pd.DataFrame.from_dict(output_data, orient='index')
        output_df = output_df.transpose()
        output_df.set_index('Date', inplace=True)
        output_df.index = output_df.index.normalize()
        output_df.to_csv('..\\result\\Forecast_Result.csv')

        output_interval = {'PI_Ih' : PI_Ih,
                           'PI_Iicu': PI_Iicu,
                           'PI_Iv' : PI_Iv}
        output_interval_df = pd.DataFrame.from_dict(output_interval, orient='index')
        output_interval_df.to_csv('..\\result\\Prediction_Interval.csv')

        upper_PI_Ih = numpy.zeros(len(solution[:, 3]))
        lower_PI_Ih = numpy.zeros(len(solution[:, 3]))
        upper_PI_Iicu = numpy.zeros(len(solution[:, 4]))
        lower_PI_Iicu = numpy.zeros(len(solution[:, 4]))
        upper_PI_Iv = numpy.zeros(len(solution[:, 5]))
        lower_PI_Iv = numpy.zeros(len(solution[:, 5]))

        for i in range(len(solution[:, 3])):
            if i == 0:
                upper_PI_Ih[i] = solution[:, 3][i] * hospital_market_share
                lower_PI_Ih[i] = solution[:, 3][i] * hospital_market_share
                upper_PI_Iicu[i] = solution[:, 4][i] * hospital_market_share
                lower_PI_Iicu[i] = solution[:, 4][i] * hospital_market_share
                upper_PI_Iv[i] = solution[:, 5][i] * hospital_market_share
                lower_PI_Iv[i] = solution[:, 5][i] * hospital_market_share
            else:
                upper_PI_Ih[i] = solution[:, 3][i] * hospital_market_share + PI_Ih
                lower_PI_Ih[i] = solution[:, 3][i] * hospital_market_share - PI_Ih
                upper_PI_Iicu[i] = solution[:, 4][i] * hospital_market_share + PI_Iicu
                lower_PI_Iicu[i] = solution[:, 4][i] * hospital_market_share - PI_Iicu
                upper_PI_Iv[i] = solution[:, 5][i] * hospital_market_share + PI_Iv
                lower_PI_Iv[i] = solution[:, 5][i] * hospital_market_share - PI_Iv

        plt.figure(dpi=1200)
        plt.figure(figsize=[16, 4])
        ax = plt.subplot(111)
        # fix
        # solution[:, 3][0] = 94 / hospital_market_share
        # solution[:, 3][1] = 100 / hospital_market_share
        # solution[:, 3][2] = 110 / hospital_market_share
        # solution[:, 3][3] = 115 / hospital_market_share
        #
        ax.plot(date, solution[:, 3] * hospital_market_share, label="Number of All Beds Forecast(t)", color='navy')
        ax.plot(date_real, realdata, label="Number of Current All Beds In Use", linestyle="--", marker=".",
                color='royalblue')

        ax.plot(date, solution[:, 4] * hospital_market_share, label="Number of ICU Beds Forecast(t)", color='tomato')
        ax.plot(date_real, real_icu, label="Number of Current Icu Beds In Use", linestyle="--", marker=".",
                color='lightsalmon')

        # ax.plot(date, solution[:, 5] * hospital_market_share, label="Number of Ventilators Forecast(t)", color='green')
        # ax.plot(date_real, real_vent, label="Number of Current Ventilators In Use", linestyle="--", marker=".",
        #         color='yellowgreen')

        p_Ih = ax.fill_between(date, upper_PI_Ih, lower_PI_Ih, color="Blue", linestyle="--", alpha=0.1)
        # p_Iicu = ax.fill_between(date, upper_PI_Iicu, lower_PI_Iicu, color="R", linestyle="--", alpha = 0.1)
        # p_Iv = ax.fill_between(date, upper_PI_Iv, lower_PI_Iv, color="G", linestyle="--", alpha = 0.1)

        ax.grid()
        box = ax.get_position()
        # Put a legend below current axis
        l1 = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3)
        l2 = ax.legend([p_Ih], ['95% Prediction Interval'])
        gca().add_artist(l1)
        # plt.legend(loc = "upper right")
        plt.xlabel("Time")
        plt.ylabel("Proportions")
        plt.title("SEIR+MHD Model (West Kendall) Forecast")
        plt.savefig('..\\result\\SEIR+MHD graph.png', dpi=800, bbox_extra_artists=(l1, l2),
                    bbox_inches='tight')
