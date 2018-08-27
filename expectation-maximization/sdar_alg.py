# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 15:44:09 2018

@author: Gabriele
"""

import numpy as np
import copy as cp

def sdar(X, discount = 0.1, k = 10):

    n = X.shape[1]
#initialization: parameters
    mu_hat = np.zeros()
    
    C_j = np.zeros(shape = (X.shape[0], k))
    
    C_j_prev = np.zeros(shape = (X.shape[0], k))
    
    weights = np.zeros([X.shape[0] , k])
    
    A = np.zeros(k,k)
    
    X_hat = np.zeros([X.shape[0] ,n-k])
    
    gaussian_means = np.zeros([d, n-k])
    
    sigma_hat = np.zeros(shape = X.shape[0] , X.shape[0] , n-k)
    
    for i in range (k,n):
        
        mu_hat = (1 - discount) * mu_hat + discount * X[:, i]
        
        gaussian_means[:,i] = mu_hat
        
        for j in range (k):
                    
            C_j[:,j] = (1 - discount) * C_j[:,j] + discount * np.dot(X[:, i] - mu_hat, X[:, i-j] - mu_hat) 
            
        window = np.hstack([C_j_prev[:,:-1], C_j]) #stack of the overall window
        
        for j in range (X.shape[0]):
            
            Y = C_j[j,:]
            
            for wind in range (k):
                
                A[wind] = window[j, 1+wind:1+wind+k] 
                
            weights[j] = np.linalg.solve(A, Y)
            
        C_j_prev = cp.deepcopy(C_j)
    
    #ghost comparing variable           
        X_hat[:,i] = np.sum(np.dot(weights[j] , X[:, i-k:i] - mu_hat), axis = 1) + mu_hat
        
        sigma_hat[:,:,i] = (1 - discount) * sigma_hat[:,:,max(0,i-1)] + discount * np.sum((X[:,i] - X_hat[:,i])**2)
        
     #multivariate gaussian
def multivariate_gaussian(x, mean, cov):

    d = x.shape[0];
     
    # covariance matrices have positive (or null) determinant
    det = np.linalg.det(cov);
    inv = np.linalg.inv(cov);
          
    g_exp = (-1/2) * np.dot(np.dot((x-mean), inv), (x-mean));
     
    p_x = (np.exp(g_exp))/np.sqrt((2*np.pi)**d * det);
     
    return p_x;


        
        
    
        
        
            
            
    
        
        