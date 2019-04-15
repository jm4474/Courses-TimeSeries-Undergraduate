# Lecture 20-21 Bayesian Analysis

import pandas as pd
import os
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.tsatools import lagmat2ds
from scipy import stats
import matplotlib.pyplot as plt

#%% load data
os.chdir("D:/Columbia/columbia courses/Time Series 2019/L20-21")
xl = pd.ExcelFile('GDPC1.xls')

#print sheet names
print(xl.sheet_names)

# Load a sheet into a DataFrame by name: df1
df1 = xl.parse('FRED Graph')

# set up dates
date = df1['Date'] # input the header of the column
print(date) # python can automatically recognize the datatype as datetime
type(date)
date = np.array(date)

# set up GDP time series
gdp = df1['GDPC1']
print(gdp)
type(gdp)
gdp = np.array(gdp)

# Y = 400 * delta log GDP
Y = 400 * (np.log(gdp[1:])-np.log(gdp[0:-1]))

#%%  AR(1) MLE arima package in python
ar1_model         = sm.tsa.ARMA(Y, (1, 0)).fit(trend='c', disp=0)

mu_arima_esti     = ar1_model.params[0]

phi_arima_esti    = ar1_model.params[1]

mu_arima_CI       = ar1_model.conf_int()[0] # extract confidence interval 95%

phi_arima_CI      = ar1_model.conf_int()[1] 

sigma2_arima_esti = ar1_model.sigma2

llf_arima_esti    = ar1_model.llf

print(ar1_model.summary())

mu_arima_results = [mu_arima_esti, mu_arima_CI]
phi_arima_results = [phi_arima_esti, phi_arima_CI]
pd.DataFrame({'mu': mu_arima_results,'phi':phi_arima_results}, index=['ML estimation', '95% condifence interval'])

#%% OLS estimation of the AR(1) parameters
Y_endog = Y[1:]

Ylag    = np.transpose(np.matrix(lagmat2ds(x=Y, maxlag0=1)[1:,1])) # exclude the first missing point
                                                                   # convert into matrix to match the datatype of mu_aux in order for concatenation
mu_aux  = np.transpose(np.matrix(np.ones(len(Ylag))))

exogen  = np.array(np.concatenate((mu_aux, Ylag), axis=1))

OLS_reg = sm.OLS(endog=Y_endog, exog=exogen)

results = OLS_reg.fit()

print(results.summary())

mu_OLS     = results.params[0]
phi_OLS    = results.params[1]
mu_OLS_CI  = results.conf_int()[0]
phi_OLS_CI = results.conf_int()[1]
sigma2_OLS = results.scale

mu_OLS_results  = [mu_OLS, mu_OLS_CI]
phi_OLS_results = [phi_OLS, phi_OLS_CI]
pd.DataFrame({'mu': mu_OLS_results,'phi':phi_OLS_results}, index=['OLS estimation', '95% condifence interval'])

#%% Bayesian Estimation of the AR(1) parameters 

# Define a class to store values in Bayes Analysis
class BayesAnalysis(object):
    pass

Bayes = BayesAnalysis()

# Guassian Prior
Bayes.Yreg = np.transpose(np.matrix(Y[1:]))

x_aux1     = np.transpose(np.matrix(np.ones(len(Bayes.Yreg)))) # x_aux1 is an n by 1 matrix

x_aux2     = np.transpose(np.matrix(Y[0:-1]))

Bayes.Xreg = np.concatenate((x_aux1, x_aux2), axis=1)

# Prior Hyperparameter
Bayes.V = np.matrix(np.identity(np.shape(Bayes.Xreg)[1])) # creat an identity matrix

#Bayes.*?
dir(Bayes)


#%% The mean of the posterior distribution is:
# we take the ML estimation of sigma2 from arma package as that in the prior
# we assume mu = 0 in the prior
sigma2 = sigma2_arima_esti

Bayes.PostMean = np.linalg.inv(np.linalg.inv(sigma2*Bayes.V) + np.transpose(Bayes.Xreg)*Bayes.Xreg)*\
(np.transpose(Bayes.Xreg)*Bayes.Yreg)

Bayes.mu_hat = Bayes.PostMean[0,0]
Bayes.phi_hat = Bayes.PostMean[1,0]

pd.DataFrame({'Bayes estimation':[Bayes.mu_hat,Bayes.phi_hat]}, index=(["mu","phi"]))

#%% An alternative way to set the prior
# we take the OLS estimation of sigma2 in prior
# we take the OLS estimation of  mu in prior
sigma2 = sigma2_OLS
mu     = mu_OLS

Bayes.PostMean = np.linalg.inv(np.linalg.inv(sigma2*Bayes.V) + np.transpose(Bayes.Xreg)*Bayes.Xreg)*\
(np.transpose(Bayes.Xreg)*Bayes.Yreg + np.linalg.inv(sigma2*Bayes.V)*mu*np.transpose(np.matrix(np.ones(np.shape(Bayes.Xreg)[1]))))

Bayes.mu_hat = Bayes.PostMean[0,0]
Bayes.phi_hat = Bayes.PostMean[1,0]

pd.DataFrame({'Bayes estimation':[Bayes.mu_hat,Bayes.phi_hat]}, index=(["mu","phi"]))

#%% The Variance of the posterior distribution is:
Bayes.PostVar = sigma2*np.linalg.inv((np.linalg.inv(Bayes.V) + np.transpose(Bayes.Xreg)*Bayes.Xreg))

#%% Compare the Prior for mu vs the posterior for mu
mu_lim = np.linspace(-3, 3, 100)

mu_prior_pdf = stats.norm.pdf(mu_lim,loc=0, scale= sigma2**0.5)

mu_post_pdf  = stats.norm.pdf(mu_lim,loc= Bayes.mu_hat, scale= Bayes.PostVar[0,0]**0.5)

plt.plot(mu_lim, mu_prior_pdf, 'r--', lw=2, alpha=0.6)

plt.plot(mu_lim, mu_post_pdf, 'b-', lw=2, alpha=0.6)   

plt.legend(['prior pdf','posterior pdf'], loc='upper left')

plt.title("Prior/Posterior of $\mu$")


#%% Compare the Prior for phi vs the posterior for phi
phi_lim = np.linspace(-3, 3, 100)

phi_prior_pdf = stats.norm.pdf(phi_lim,loc=0, scale= sigma2**0.5)

phi_post_pdf  = stats.norm.pdf(phi_lim,loc= Bayes.phi_hat, scale= Bayes.PostVar[1,1]**0.5)

plt.plot(phi_lim, phi_prior_pdf, 'r--', lw=2, alpha=0.6)

plt.plot(phi_lim, phi_post_pdf, 'b-', lw=2, alpha=0.6)   

plt.legend(['prior pdf','posterior pdf'], loc='upper left')

plt.title("Prior/Posterior of $\phi$")

#%% 95% Credible Sets based on the quantiles of the posterior

# Standard Normal inverse cumulative distribution function (percent point function)
Bayes.mu_CS = [stats.norm.ppf(0.025, loc=Bayes.mu_hat, scale=Bayes.PostVar[0,0]**0.5),\
               stats.norm.ppf(0.975, loc=Bayes.mu_hat, scale=Bayes.PostVar[0,0]**0.5)]

Bayes.phi_CS = [stats.norm.ppf(0.025, loc=Bayes.phi_hat, scale=Bayes.PostVar[1,1]**0.5),\
               stats.norm.ppf(0.975, loc=Bayes.phi_hat, scale=Bayes.PostVar[1,1]**0.5)]

print('The 95% credible set for mu and phi are', Bayes.mu_CS, "and", Bayes.phi_CS,".")

#%% Compare MLE, OLS and Bayesian Analysis
# change the precision of confidence interval
mu_arima_CI[0]  = round(mu_arima_CI[0], 5) 
mu_arima_CI[1]  = round(mu_arima_CI[1], 5)
phi_arima_CI[0] = round(phi_arima_CI[0], 5) 
phi_arima_CI[1] = round(phi_arima_CI[1], 5)
mu_OLS_CI[0]    = round(mu_OLS_CI[0], 5) 
mu_OLS_CI[1]    = round(mu_OLS_CI[1], 5)
phi_OLS_CI[0]   = round(phi_OLS_CI[0], 5) 
phi_OLS_CI[1]   = round(phi_OLS_CI[1], 5)
Bayes.mu_CS[0]  = round(Bayes.mu_CS[0], 5) 
Bayes.mu_CS[1]  = round(Bayes.mu_CS[1], 5)
Bayes.phi_CS[0] = round(Bayes.phi_CS[0], 5) 
Bayes.phi_CS[1] = round(Bayes.phi_CS[1], 5)

arima_results = [mu_arima_esti, mu_arima_CI, phi_arima_esti, phi_arima_CI]

OLS_results = [mu_OLS, mu_OLS_CI, phi_OLS, phi_OLS_CI]

Bayes.results = [Bayes.mu_hat, Bayes.mu_CS, Bayes.phi_hat, Bayes.phi_CS]

pd.DataFrame({'arma MLE':arima_results,'OLS':OLS_results,'Bayesian Analysis':Bayes.results},\
             index=['mu estimation', 'mu confidence interval/credit set', 'phi estimation', 'phi confidence interval/credit set'])
