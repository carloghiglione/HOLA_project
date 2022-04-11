# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 10:37:22 2022

@author: carlo
"""
import numpy as np

# this environment takes also prices that are associated to each arm
# I also have "conversion rate": prob of selling an item at a specific price c: price -> [0,1]
class env:
    def __init__(self, probs, prices):
        self.prices = prices     # vector of prices associated to each arm
        self.probs = probs       # vector of probabilities of a Bernoulli for each arm, the observed data are {0,1}
        
     
    # simulate from bernoulli    
    def round(self, arm_pulled):
        conv = np.random.binomial(n=1, p=self.probs[arm_pulled])    # conversion probability of the pulled arm
        reward = conv*self.prices[arm_pulled]                        # reward for the pulled arm
        return reward  

# the optimal solution is to find the maximum of (price)*(prob of selling the item)   

# now the reward is not just if I sell or not the product, but the profit that I have
# that is the (probability of selling it) * (its price)
# now the support of the random variable, the observation, is [0, max_price]