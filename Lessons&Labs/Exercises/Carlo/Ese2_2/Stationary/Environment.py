# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 10:37:22 2022

@author: carlo
"""
import numpy as np


class environment:
    def __init__(self, probs):
        self.probs = probs       # vector (or matrix) of probabilities of a Bernoulli for each arm, the observed data are {0,1}
        
    # simulate from bernoulli    
    def round(self, arm_pulled):
        reward = np.random.binomial(n=1, p=self.probs[arm_pulled])    # simulate an observation from the pulled arm
        return reward  

# the optimal solution is to find the maximum of (price)*(prob of selling the item)   
