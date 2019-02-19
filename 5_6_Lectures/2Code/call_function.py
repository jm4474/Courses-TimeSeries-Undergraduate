

# call a function
import pandas as pd

from Gaussian_MA_q import GaussianMAq

MA_output = GaussianMAq(theta = [1,0.5,0.3,0.8],sigma = 1, H = 4, I = 1000)

acf_H     = MA_output[0]

MC_SD     = MA_output[1]

print(pd.DataFrame(acf_H,columns={"autocovariance function"}))