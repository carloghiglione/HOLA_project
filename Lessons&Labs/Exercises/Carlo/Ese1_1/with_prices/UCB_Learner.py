# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 11:13:41 2022

@author: carlo
"""
import numpy as np
from Learner import learner


class ucb(learner):
    
    def __init__(self, n_arms, prices):
        super().__init__(n_arms)
        self.means = np.zeros(n_arms)                            # to collect means
        self.widths = np.array([np.inf for _ in range(n_arms)])  # to collect upper confidence bounds (- mean)
        self.prices = prices                                     # I know the set of prices, not the conversion rates
        
    # to select the arms with highest upper confidence bound
    def act(self):
        idx = np.argmax((self.means + self.widths)*self.prices)  # I multiply everything by the prices, then returnt the max
        return idx
    
    def update(self, arm_pulled, reward):
        reward = reward > 0                  # I check if I have sold the product
        super().update(arm_pulled, reward)
        self.means[arm_pulled] = np.mean(self.reward_per_arm[arm_pulled])    # update the mean of the arm we pulled
        for idx in range(self.n_arms):                                       # for all arms, update upper confidence bounds
            n = len(self.reward_per_arm[idx])
            if n > 0:
                self.widths[idx] = np.sqrt(2*np.log(self.t)/n)
            else:
                self.widths[idx] = np.inf
