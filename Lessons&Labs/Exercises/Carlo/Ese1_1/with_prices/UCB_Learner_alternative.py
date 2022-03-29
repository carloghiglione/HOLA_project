# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 12:32:06 2022

@author: carlo
"""
import numpy as np
from Learner import learner


# second approach
# directly estimate prices*conversion rate that is what I want to optimize
class ucb2(learner):
    
    def __init__(self, n_arms, prices):
        super().__init__(n_arms)
        self.means = np.zeros(n_arms)                            # to collect means
        self.widths = np.array([np.inf for _ in range(n_arms)])  # to collect upper confidence bounds (- mean)
        self.prices = prices                                     # I know the set of prices, not the conversion rates
        
    # to select the arms with highest upper confidence bound
    def act(self):
        idx = np.argmax(self.means + self.widths) 
        return idx
    
    def update(self, arm_pulled, reward):
        super().update(arm_pulled, reward)
        self.means[arm_pulled] = np.mean(self.reward_per_arm[arm_pulled])    # update the mean of the arm we pulled
        for idx in range(self.n_arms):                                       # for all arms, update upper confidence bounds
            n = len(self.reward_per_arm[idx])
            if n > 0:
                self.widths[idx] = np.sqrt(2*np.max(self.prices)*np.log(self.t)/n)   # now the support is [0, max_price] 
            else:
                self.widths[idx] = np.inf
