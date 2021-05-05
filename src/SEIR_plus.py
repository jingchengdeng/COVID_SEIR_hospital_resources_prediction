# -*- coding: utf-8 -*-
"""
SEIR_plus.py
Created on 2020/5/5

@author: yiqing
"""

from train import Train
import pandas as pd
import numpy as np


def read_data():
    """
    :return:
    """
    data = pd.read_csv('..\\data\\spara_data\\csv\\DataWestKendall.csv', index_col=[0])
    data = data.fillna(0)
    pos = data['positive_IP_count']
    discharge = data['pos_discharged_count']
    deceased = data['pos_deceased_count']
    cum = cumulative(pos, discharge, deceased)
    data['cumulative_pos_IP'] = cum
    for i in range(len(data)):
        if data['Date'][i] != '2020-10-01':
            data = data.drop([i])
        else:
            break

    return data


def cumulative(pos, discharge, deceased):
    """
    :param pos:
    :param discharge:
    :param deceased:
    :return:
    """
    size = len(pos)
    cum = np.zeros(size)
    for i in range(size):
        if i == 0:
            cum[i] = pos[i] - discharge[i] - deceased[i]
        else:
            cum[i] = cum[i - 1] + pos[i] - discharge[i] - deceased[i]
    return cum


def read_param():
    para_df = pd.read_csv('..\\data\\spara_data\\csv\\ParamWestKendall.csv', index_col=[0])
    predict_Days = para_df.loc['Predict_Days'].values[0] + 136
    hospital_market_share = para_df.loc['Hospital_Market_Share'].values[0]
    population = para_df.loc['Population'].values[0]
    infected = para_df.loc['Infected'].values[0]
    death = para_df.loc['Death'].values[0]
    recovered = para_df.loc['Recovered'].values[0]
    number_of_current_hospitalized_patients = para_df.loc['Hospitalized_Patient'].values[0]
    hospitalization_percent = para_df.loc['Hospitalized_Percentage'].values[0]
    hospital_stay = para_df.loc['Hospital_Average_Stay'].values[0]
    ICU_rate = 0.015 #para_df.loc['ICU_Percentage'].values[0]
    ICU_stay = para_df.loc['ICU_Average_Stay'].values[0]
    ventilated_percent = para_df.loc['Ventilator_Percentage'].values[0]
    ventilator_days = para_df.loc['Ventilator_Average_Days'].values[0]
    double_time = para_df.loc['Double_time'].values[0]
    duration_of_immunization = para_df.loc['Duration_of_Immunization'].values[0]
    death_rate = para_df.loc['Death_Rate'].values[0]
    mean_latent_period = para_df.loc['Mean_Latent_Period'].values[0]
    mean_recovery_time = para_df.loc['Mean_recovery_time'].values[0]
    social_distancing = para_df.loc['Social_Distance'].values[0]

    return predict_Days, hospital_market_share, population, infected, death, recovered, \
           number_of_current_hospitalized_patients, hospitalization_percent, hospital_stay, ICU_rate, ICU_stay, \
           ventilated_percent, ventilator_days, double_time, duration_of_immunization, death_rate, \
           mean_latent_period, mean_recovery_time, social_distancing


def set_param(real_days, realdata, real_icu, real_vent):

    predict_Days, hospital_market_share, population, infected, death, recovered, \
    number_of_current_hospitalized_patients, hospitalization_percent, hospital_stay, ICU_rate, ICU_stay, \
    ventilated_percent, ventilator_days, double_time, duration_of_immunization, death_rate, \
    mean_latent_period, mean_recovery_time, social_distancing = read_param()

    N = population
    I = infected
    E = I * mean_latent_period
    R = 0
    D = death
    S = N - E - I - R - D
    Ih = number_of_current_hospitalized_patients / hospital_market_share
    Im = I - Ih
    Iicu = 3 / hospital_market_share
    Iv = 0 / hospital_market_share

    predict_len = predict_Days
    real_len = real_days
    seir = [S, E, Im, Ih, Iicu, Iv, D, R]

    # calculate beta
    rate_of_growth = 2 ** (1.0 / double_time) - 1.0
    beta1 = (rate_of_growth + 1 / mean_recovery_time) / S * (1 - social_distancing)
    beta2 = beta1

    # parameters
    sigma = 1 / duration_of_immunization
    alpha = 1 / mean_latent_period
    rho = hospitalization_percent
    rho_icu = ICU_rate
    rho_v = ventilated_percent
    lamda1 = 1 / mean_recovery_time
    lamda2 = 1 / hospital_stay
    lamda_icu = 1 / ICU_stay
    lamda_v = 1 / ventilator_days
    kappa = death_rate

    # pack parameters
    '''
        predict_len, real_len, realdata, real_icu, real_vent, seir, N, beta1, beta2, sigma, alpha, rho,
                     rho_icu, rho_v,
                     lamda1, lamda2, lamda_icu, lamda_v, kappa, hospital_market_share
    '''
    param = [predict_len, real_len, realdata, real_icu, real_vent, seir, N, beta1, beta2, sigma, alpha, rho,
             rho_icu, rho_v, lamda1, lamda2, lamda_icu, lamda_v, kappa, hospital_market_share]
    return param


def main():
    data = read_data()
    real_Ih = data['cumulative_pos_IP']
    real_icu = data['current_icu']
    real_vent = data['current_vent']
    real_days = len(real_Ih)
    param = set_param(real_days, real_Ih, real_icu, real_vent)
    date = pd.date_range(start='2020-10-01', periods=param[0])
    date_real = pd.date_range(start='2020-10-01', periods=param[1])
    model = Train(*param)
    model.output(date, date_real, param[-1], real_Ih, real_icu, real_vent)


if __name__ == '__main__':
    main()
