#!/usr/bin/env python
# coding: utf-8

# In[1]:


#####################################
import sys
sys.path.append("..\\src")
import numpy
import matplotlib.pyplot as plt
import pandas as pd
from SIR_model import SIR
from SEIR_model import SEIR
from SEIR_plusCAQ_model import SEIR_plusCAQ
from SEIR_plusCQD_model import SEIR_plusCQD
from SEIR_plusBG_model import SEIR_plusBG
from SEIR_plus_model import SEIR_plus


# In[2]:


TS_data_confirm = pd.read_csv("..\\data\\jh_data\\csse_covid_19_data\\csse_covid_19_time_series\\time_series_covid19_confirmed_US.csv")
TS_data_death = pd.read_csv("..\\data\\jh_data\\csse_covid_19_data\\csse_covid_19_time_series\\time_series_covid19_deaths_US.csv") 
TS_data_confirm.set_index('UID',inplace = True)
TS_data_death.set_index('UID',inplace = True)
Miami_dade_comfirm = TS_data_confirm.loc[84012086]
Miami_dade_death = TS_data_death.loc[84012086]
Miami_dade_comfirm = Miami_dade_comfirm.drop(labels=['iso2', 'iso3','code3','FIPS','Province_State','Country_Region','Lat','Long_','Combined_Key','Admin2'])
Miami_dade_comfirm = Miami_dade_comfirm[Miami_dade_comfirm!=0]
Miami_dade_death = Miami_dade_death.drop(labels=['iso2', 'iso3','code3','FIPS','Province_State','Country_Region','Lat','Long_','Combined_Key','Admin2'])
for i in range(len(Miami_dade_death)):
    if Miami_dade_death.index[0] != Miami_dade_comfirm.index[0]:
        Miami_dade_death = Miami_dade_death.drop([Miami_dade_death.index[0]])
BHSF_data = pd.read_excel("..\data\Ih_D.xlsx")
BHSF_data = BHSF_data.fillna(0)
BHSF_data.head(50)
#print(Miami_dade_comfirm)
#print(Miami_dade_death)
# real_data = BHSF_data['cumulative_pos_IP'].to_numpy()
# real_data = numpy.delete(real_data, [0,1,2,3,4])
# print(real_data)


# In[3]:


# init parameters
mean_latent_period = 5
mean_recovery_time = 14
double_time = 5.5
rate_of_growth = 2**(1.0/double_time) - 1.0

population = 2716940         # init N
number_of_current_hospitalized_patients = 50
infected = 278
hospital_market_share = 0.22          # % of people will come to your hospital
hospitalization_percent= (135/hospital_market_share)/7712
#number_of_all_infected = 
social_distancing = 0              # %
ICU_rate = (135/532)*hospital_market_share
ventilated_percent = (85/532)*hospital_market_share
hospital_stay = 7
ICU_stay = 9.1
Ventilator_days = 11.6
duration_of_immunization = 60
death_rate = 0.022
t=numpy.linspace(0,80,80)
# N, beta1, beta2, sigma, alpha, rho, rho_icu, rho_v,
# lamda1, lamda2, lamda_icu, lamda_v, kappa
N = population #population
I = infected   #(number_of_current_hospitalized_patients / hospital_market_share)/hospitalization_percent
E = I * mean_latent_period
R = 0
D = 7
S = N - E - I - R - D
Ih = number_of_current_hospitalized_patients/hospital_market_share
Im = I - Ih

Iicu = 15/hospital_market_share
Iv = 6/hospital_market_share
beta = (rate_of_growth + 1/mean_recovery_time)/S * (1- social_distancing)

date = pd.date_range(start="2020-03-23",end="2020-06-11",periods=80)
date = numpy.array(date)
date_real = pd.date_range(start="2020-03-23",end="2020-04-19",periods=28)
date_real = numpy.array(date_real)


# In[4]:


#SEIR+MHD
# param
beta1 = beta
beta2 = beta1 * 0.1
sigma = 1/duration_of_immunization
alpha = 1/mean_latent_period
rho = hospitalization_percent
rho_icu = ICU_rate
rho_v = ventilated_percent
lamda1 = 1/ mean_recovery_time
lamda2 = 1/ hospital_stay
lamda_icu = 1/ ICU_stay
lamda_v = 1/Ventilator_days
kappa = death_rate
data = [S, E, Im, Ih, Iicu, Iv, D, R]
SEIR_MHD = SEIR_plus(data, t, N, beta1, beta2, sigma, alpha, rho, rho_icu, rho_v,
                           lamda1, lamda2, lamda_icu, lamda_v, kappa)
solution = SEIR_MHD.solve()


# In[5]:


#realdata = numpy.array([155,168,166,167,166,170,155,160,156,151,147,145,135,135])
realdata = BHSF_data['cumulative_pos_IP'].to_numpy()
realdata = numpy.delete(realdata, [0,1,2,3,4,5,6,7,8,9,10,11])
real_vent = numpy.array([6,13,15,21,27,29,31,30,32,34,36,41,42,43,48,50,48,45,47,49,48,47,47,49,47,45,39,39,37,33])
real_icu = numpy.array([15,19,18,25,31,40,41,40,41,47,50,58,60,60,65,66,61,62,70,65,65,67,61,61,62,62,61,56,56,57])
print(real_vent.shape)


# In[6]:


beta1, beta2, alpha ,lamda1, rho, kappa, sigma= SEIR_MHD.train(realdata/hospital_market_share)
print(beta1,beta2,alpha,lamda1, rho, kappa, sigma)


# In[7]:


SEIR_MHD = SEIR_plus(data, t, N, beta1, beta2, sigma, alpha, rho, rho_icu, rho_v,
                           lamda1, lamda2, lamda_icu, lamda_v, kappa)
solution = SEIR_MHD.solve()


# In[8]:


date_real = pd.date_range(start="2020-03-23",end="2020-04-21",periods=30)
date_real = numpy.array(date_real)
realdata = numpy.append(realdata,106)
realdata = numpy.append(realdata,103)
size  = len(realdata)
RMSE_Ih= numpy.sqrt(numpy.mean((solution[:,3][:30]*hospital_market_share - realdata) ** 2))
PI_Ih = 1.96*RMSE_Ih
RMSE_Iicu = numpy.sqrt(numpy.mean((solution[:,4][:30]*hospital_market_share - real_icu) ** 2))
PI_Iicu = 1.96*RMSE_Iicu
RMSE_Iv = numpy.sqrt(numpy.mean((solution[:,5][:30]*hospital_market_share - real_vent) ** 2))
PI_Iv = 1.96*RMSE_Iv

print(PI_Ih,PI_Iicu,PI_Iv)
upper_PI_Ih= numpy.zeros(len(solution[:,3]))
lower_PI_Ih= numpy.zeros(len(solution[:,3]))
upper_PI_Iicu= numpy.zeros(len(solution[:,4]))
lower_PI_Iicu= numpy.zeros(len(solution[:,4]))
upper_PI_Iv= numpy.zeros(len(solution[:,5]))
lower_PI_Iv= numpy.zeros(len(solution[:,5]))
for i in range(len(solution[:,3])):
    if i ==0:
        upper_PI_Ih[i] = solution[:,3][i]*hospital_market_share
        lower_PI_Ih[i] = solution[:,3][i]*hospital_market_share
        upper_PI_Iicu[i] = solution[:,4][i]*hospital_market_share
        lower_PI_Iicu[i] = solution[:,4][i]*hospital_market_share
        upper_PI_Iv[i] = solution[:,5][i]*hospital_market_share
        lower_PI_Iv[i] = solution[:,5][i]*hospital_market_share
    else:
        upper_PI_Ih[i] = solution[:,3][i]*hospital_market_share + PI_Ih
        lower_PI_Ih[i] = solution[:,3][i]*hospital_market_share - PI_Ih
        upper_PI_Iicu[i] = solution[:,4][i]*hospital_market_share + PI_Iicu
        lower_PI_Iicu[i] = solution[:,4][i]*hospital_market_share - PI_Iicu      
        upper_PI_Iv[i] = solution[:,5][i]*hospital_market_share + PI_Iv
        lower_PI_Iv[i] = solution[:,5][i]*hospital_market_share - PI_Iv     
               


# In[9]:


from matplotlib.pyplot import *
plt.figure(figsize=[16,4])
ax = plt.subplot(111)

ax.plot(date,solution[:,3]*hospital_market_share,label="Number of All Beds Forecast(t)",color = 'navy')
ax.plot(date_real,realdata,label="Number of Current All Beds In Use",linestyle="--",  marker = ".",color = 'royalblue')

ax.plot(date,solution[:,4]*hospital_market_share,label="Number of ICU Beds Forecast(t)",color = 'tomato')
ax.plot(date_real,real_icu,label="Number of Current Icu Beds In Use",linestyle="--", marker = ".",color = 'lightsalmon')

ax.plot(date,solution[:,5]*hospital_market_share,label="Number of Ventilators Forecast(t)",color = 'green')
ax.plot(date_real,real_vent,label="Number of Current Ventilators In Use",linestyle="--", marker = ".",color = 'yellowgreen')

p_Ih = ax.fill_between(date, upper_PI_Ih, lower_PI_Ih, color="B", linestyle="--", alpha = 0.1)
# p_Iicu = ax.fill_between(date, upper_PI_Iicu, lower_PI_Iicu, color="R", linestyle="--", alpha = 0.1)
# p_Iv = ax.fill_between(date, upper_PI_Iv, lower_PI_Iv, color="G", linestyle="--", alpha = 0.1)

ax.grid()
box = ax.get_position()
# Put a legend below current axis
l1 = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3)
l2 = ax.legend([p_Ih],['95% Prediction Interval'])
gca().add_artist(l1)
#plt.legend(loc = "upper right")
plt.xlabel("Time")
plt.ylabel("Proportions")
plt.title("SEIR+MHD Model Forecast with Validation of Ih (BHSF) 04/22")
plt.show()


# In[10]:


output_data = {'Date':date,
               'Number of Current All Beds In Use': realdata,
               'Number of All Beds Forecast': numpy.ceil(solution[:,3]*hospital_market_share),
               'Number of Current Icu Beds In Use': real_icu,
               'Number of ICU Beds Forecast': numpy.ceil(solution[:,4]*hospital_market_share),
               'Number of Current Ventilators In Use': real_vent,
               'Number of Ventilators Forecast': numpy.ceil(solution[:,5]*hospital_market_share)}

output_df = pd.DataFrame.from_dict(output_data, orient='index')
output_df = output_df.transpose()
output_df.set_index('Date',inplace = True)
output_df.index = output_df.index.normalize()
output_df.to_csv('..\\result\\forcast_result_04_22.csv')


# In[ ]:





# In[ ]:




