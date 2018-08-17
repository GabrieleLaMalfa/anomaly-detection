# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 15:17:58 2018

@author: Gabriele

Sequentially discounting expectation maximization algorithm 
"""
import numpy as np
import math 

def SDEM(X, n_mixtures, init_method = "uniform", alpha = 1, discount = 1):
    
    centers = np.array([])    
    
    for _ in range (n_mixtures):
        
        for i in range (X.shape[0]):

            centers = np.append(centers, 
                                np.random.uniform(X[i].min() , X[i].max()))

    centers = centers.reshape(n_mixtures, X.shape[0])
    
        
    
    ##initialize covariance matrices (isotropic)
    
    cov = np.zeros(shape=(n_mixtures, X.shape[0], X.shape[0]))    
    
    for i in range (n_mixtures):
        cov[i] = np.eye(X.shape[0])
        
    ##initialize weights of each gaussian distrib.
    
    weights = np.array([1/n_mixtures for _ in range(n_mixtures)])
    
    #gamma initialization
    
    gamma = np.zeros(n_mixtures)
        
    #expectation
    
    for i in range (X.shape[1]):
        
        for j in range (n_mixtures):
            
            tot_prob = np.sum([weights[n] * multivariate_gaussian(X[:,i], centers[n], cov[n]) for n in range (n_mixtures)])
        
            p = multivariate_gaussian(X[:,i], centers[j], cov[j])
            
            gamma[j] = (1 - alpha * discount) * p/tot_prob + (alpha * discount) / n_mixtures
        
#probability function: Multivariate Gaussian
    
def multivariate_gaussian(x, mean, cov):
    
    d = x.shape[0]
    
    det_cov = np.linalg.det(cov)
    
    inv_cov = np.linalg.inv(cov)
    
    g_exp = -1/2 * np.dot(np.dot(x - mean, inv_cov), x - mean)
    
    p_x = (1/np.power(2 * np.pi, d/2) * np.sqrt(det_cov) * np.exp(g_exp))
    
    return p_x
    
    
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            