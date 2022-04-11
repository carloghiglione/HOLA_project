# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 14:46:01 2022

@author: carlo
"""
import numpy as np

# also requires the time horizon of the observation
# I need prob for each arm and for each period (a matrix of probs)
class non_stat_env:
    
    def __init__(self, n_arms, probs_matrix, horizon):
        self.n_arms = n_arms
        self.probs_matrix = probs_matrix
        self.horizon = horizon
        self.n_changes = len(probs_matrix)    # number of phases
        self.t = 0
        self.inner_horizon = self.horizon // self.n_changes  # length of a phase (does division, then int)
        self.phase = 0                        # current phase I am
        
    def round(self, pulled_arm):
        if self.t > (self.phase+1)*self.inner_horizon:    # increment the phase if I change it
            self.phase = min(self.phase+1, self.n_changes-1)
        reward = np.random.binomial(n=1, p=self.probs_matrix[self.phase, pulled_arm])
        self.t += 1
        return reward