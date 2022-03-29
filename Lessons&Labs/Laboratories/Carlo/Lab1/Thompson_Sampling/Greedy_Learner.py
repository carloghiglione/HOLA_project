# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 15:13:48 2022

@author: carlo
"""
from Learner import *

class GR_Learner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.expected_rewards = np.zeros(n_arms)
        
    # to select the arm to pull
    def pull_arm(self):
        if(self.t < self.n_arms):
            return self.t
        idxs = np.argwhere(self.expected_rewards == self.expected_rewards.max()).reshape(-1)
        pulled_arm = np.random.choice(idxs)
        return pulled_arm
    
    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.expected_rewards[pulled_arm] = (self.expected_rewards[pulled_arm]*(self.t - 1) + reward) / self.t       