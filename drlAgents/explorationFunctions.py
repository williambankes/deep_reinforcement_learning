# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 21:27:41 2020

@author: William Bankes
"""

#%%

from .logger import Logger

import math
import numpy as np

#%%

def create_exponential_exploration(eps_start=0.9, eps_end=0.05, eps_decay=1):
    """
    creates the exponential exploration function with parameters:
        
        eps_start -> (double) value returned at time 0
        
        eps_end -> (double) asymptotic value as time goes to infinity
        
        eps_decay -> (double) the rate of exponential decay
    """
    
    def exponential_exploration(t):
    
        eps_threshold = eps_end + (eps_start - eps_end)* math.exp(-1 * t/eps_decay)
        Logger.getInstance().add('ep_threshold', eps_threshold)
        
        return eps_threshold
        
    return exponential_exploration
        

def create_factor_min_exploration(eps_start=0.9, eps_decay=0.99, eps_end=0.01):
    
    def factor_min_exploration(t):
        
        factor = eps_decay ** t
        eps_threshold = max(eps_start * factor, eps_end)
        
        Logger.getInstance().add('ep_threshold', eps_threshold)
        
        return eps_threshold
    
    return factor_min_exploration

#%%
        
