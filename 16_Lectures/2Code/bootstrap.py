# Lecture 13-14 Bootstrap

# Set up parameters
import sys
sys.path.append('D:/Columbia/columbia courses/Time Series 2019/L13_14')

from AR1_param import AR1

#%% Simulate data sets to implement the parametric "Bootstrap" routine
#(1000 repetitions)

import numpy as np
import statsmodels.api as sm
import time

T = 100
I = 1000
ar_param = np.array([1, -AR1.phi])
ma_param = np.array([1])

Xsim       = np.zeros([T,I])
phi_OLS    = np.zeros(I)
sigma2_OLS = np.zeros(I)

loop_start = time.time()
print('The parametric bootstrap is running...')

for i in range(I):   
    Xsim[:,i]     = sm.tsa.arma_generate_sample(ar=ar_param, ma=ma_param, nsample=T, sigma = AR1.sigma2**0.5)
    Xaux          = np.matrix(Xsim[0:-1,i]) # Xaux is 1 by n
    Yaux          = np.transpose(np.matrix(Xsim[1:,i])) # Yaux is n by 1
    phi_OLS[i]    = (np.linalg.inv(Xaux*np.transpose(Xaux))*Xaux*Yaux)[0,0]
    sigma2_OLS[i] = np.mean((np.array(np.transpose(Yaux))-phi_OLS[i]*np.array(Xaux))**2)

loop_end = time.time()
print("The loop took {} seconds to finish.".format(loop_end-loop_start))

#%% Plot the histogram

import matplotlib.pyplot as plt
plt.figure(1)

plt.subplot(121)
plt.hist(x=phi_OLS, bins = 20, facecolor='steelblue', alpha = 0.5)
plt.xlabel("$\phi_{OLS}$")

plt.subplot(122)
plt.hist(x=sigma2_OLS, bins = 20, facecolor='steelblue', alpha = 0.5)
plt.xlabel("$\sigma^2_{OLS}$")
plt.show()

#%% Compare the empirical distribution of phihat with the normal approximation

plt.style.use('seaborn') # pretty matplotlib plots
from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF

x = np.linspace(np.amin(phi_OLS), np.amax(phi_OLS), I)

AsyVar=(1-(AR1.phi**2))/T
norm_cdf = stats.norm.cdf(x,loc=AR1.phi, scale=AsyVar**0.5)

ecdf = ECDF(phi_OLS)
OLScdf = ecdf(x)

plt.plot(x, norm_cdf, 'r--', lw=2, alpha=0.6) # lw: line width
plt.plot(x, OLScdf, 'g-', lw=2, alpha=0.6)
plt.legend(['normal','OLS'], loc='upper left')
plt.title('CDF')
plt.hist(phi_OLS, alpha=0.3, bins = 30, cumulative= True, density = True)


#%% Confidence interval

qt_phi = np.quantile(phi_OLS, [0.025, 0.075])
qt_sigma2 = np.quantile(sigma2_OLS, [0.025, 0.075])

CI_phi   = [qt_phi[0],qt_phi[1]]
CI_sigma2 = [qt_sigma2[0],qt_sigma2[1]]

print('The bootstrap 95% confidence interval for the phi_OLS is:', CI_phi)
print('The bootstrap 95% confidence interval for the sigma2_OLS is:', CI_sigma2)


