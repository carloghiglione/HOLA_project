# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 15:38:49 2022

@author: carlo
"""
# UCB Algorithm

from Learner import learner
import numpy as np

class UCB(learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.empirical_means = np.zeros(n_arms)       # empirical means for each arm
        self.confidence = np.array([np.inf]*n_arms)   # confidence level for each arm
        
    def pull_arm(self):
        upper_conf = self.empirical_means + self.confidence                   # compute upper confidence bound
        return np.random.choice(np.where(upper_conf == upper_conf.max())[0])  # return maximum upper confidence bound (if many max, select one max randomly)
    
    def update(self, pull_arm, reward):     # update the conficence level in each arm
        self.t += 1
        self.empirical_means[pull_arm] = (self.empirical_means[pull_arm]*(self.t - 1) + reward)/self.t   # update empirical mean for the pulled arm
        for a in range(self.n_arms):                   # for each arm, update the confidence level
            n_samples = len(self.reward_per_arm[a])    # count number of times we pulled current arm
            self.confidence[a] = (2*np.log(self.t)/n_samples)**0.5 if n_samples > 0 else np.inf  # if collected any reward for current arm, update it with formula
        self.update_observations(pull_arm, reward)
            