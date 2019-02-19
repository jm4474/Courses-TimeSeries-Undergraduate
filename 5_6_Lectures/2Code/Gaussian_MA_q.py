def GaussianMAq(theta,sigma,H,I): 
    """
    INPUTS are as follows:

    1) theta: (q+1)x1 vector of MA coefficients (or 1*(q+1), it will be converted into an array)
    
    2) sigma: Variance of the Gaussian White Noise
    
    3) H:     Largest order of the covariance function (h-th order autocovariance)
    
    4) I:     Number of Monte-Carlo draws
    
    OUTPUTS are as follows:

    1) auto_cov_H: H+1 times 1 vector containing the MC estimators of the autocovariance function
    
    2) MC_std_error: "standard error" of the MC approximation
    
    """
    #%% Generate the epsilons
    
    import numpy as np
    
    q = np.shape(theta)[0]-1 
    
    e = np.random.normal(0, sigma, [q+H+1,I])
    
    #I want to think about e as a matrix whose columns contain:
    #(epsilon_H, epsilon_{H-1}, ... epsilon_1, \epsilon_0, ..., epsilon_{-(q-1)})'
    
    #%% Generate the X's
    
    Xt    = np.matrix(np.zeros([H+1,I])) 
    
    theta = np.matrix(theta)
   
    for i_h in range(H+1): 
        Xt[i_h,:] = theta * e[i_h:i_h+q+1, :]
    
    #%% Compute the Autocovariance Function
    
    from scipy import stats
    
    auto_cov_H   = np.zeros(H+1) 
    
    MC_std_error = np.zeros(H+1)
    
    for ind_j in range(H+1):
        
        aux = (np.array(Xt[0,:])-np.mean(Xt[0,:]) ) * (np.array(Xt[ind_j,:])-np.mean(Xt[ind_j,:]) )
    
        auto_cov_H[ind_j]   \ 
            = np.mean(aux)
        
        MC_std_error[ind_j] \
            = stats.sem(aux,ddof=0,axis = None) 
    
        del aux 
        
    #%% Plot 95% confidence interval of simulated covariance
    
    import matplotlib.pyplot as plt
    
    plt.plot(np.arange(0,H+1) , auto_cov_H, linestyle='none', marker='o',markersize=7)

    plt.plot(np.arange(0,H+1) , auto_cov_H+1.96*MC_std_error, linestyle='none', marker='*', markersize=7)

    plt.plot(np.arange(0,H+1) , auto_cov_H-1.96*MC_std_error, linestyle='none', marker='*', markersize=7)

    plt.title('95% confidence interval of simulated covariance')
    
    #%% return output
    
    return auto_cov_H, MC_std_error;
    





