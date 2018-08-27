# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:36:04 2018

@author: Gabriele
"""

#Scoring

import numpy as np

def multivariate_gaussian(x, mean, cov):

    d = x.shape[0];
     
    # covariance matrices have positive (or null) determinant
    det = np.linalg.det(cov);
    inv = np.linalg.inv(cov);
          
    g_exp = (-1/2) * np.dot(np.dot((x-mean), inv), (x-mean));
     
    p_x = (np.exp(g_exp))/np.sqrt((2*np.pi)**d * det);
     
    return p_x;


def outlier_detect_gauss(x, mu, gamma):
    
    res = -np.log2(multivariate_gaussian((x, mean, cov)))
    
    
def change_point_detect(X, MU, GAMMA):
    
    t = X.shape[1] 

    tot_scoring = 0
    
    for i in range (t):
        
        tot_scoring += outlier_detect_gauss(X[:, i] , MU[:, i] , GAMMA[:,:,i])
        
    res = 1/t * tot_scoring
        
        
        
    
    
    
    
    
    