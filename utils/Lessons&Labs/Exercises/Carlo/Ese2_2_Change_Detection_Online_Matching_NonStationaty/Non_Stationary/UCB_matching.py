# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 16:05:34 2022

@author: carlo
"""
# UCB algorithm for online matching problems

from UCB import UCB
import numpy as np
from scipy.optimize import linear_sum_assignment


class UCB_Matching(UCB):
    def __init__(self, n_arms, n_rows, n_cols):   # n_cols and rows are of adiajency matrix
        super().__init__(n_arms)
        self.n_rows = n_rows
        self.n_cols = n_cols
        assert n_arms == n_cols*n_rows    # because each arm corresponds to an edge of the graph (so number of elements of adiajecy matrix), this must low
        
    def pull_arm(self):
        upper_conf = self.empirical_means + self.confidence   # compute upper confidence bound
        upper_conf[np.isinf(upper_conf)] = 1e3                # instead of use inf, use large number to make everything work
        row_ind, col_ind = linear_sum_assignment(-upper_conf.reshape(self.n_rows, self.n_cols))    # find the matching with upper confidence levels as adiajecny matrix
        return (row_ind, col_ind)      # return solution of the matching problem, the super arm
    
    def update(self, pulled_arms, rewards):
        self.t += 1
        pulled_arm_flat = np.ravel_multi_index(pulled_arms, (self.n_rows, self.n_cols))  #  convert to flat array
        for a in range(self.n_arms):                               # update confidence for each arm
            n_samples = len(self.reward_per_arm[a])
            self.confidence[a] = (2*np.log(self.t)/n_samples)**0.5 if n_samples > 0 else np.inf
        for pulled_arm, reward in zip(pulled_arm_flat, rewards):   # update observations for each arm  in the super arm
            self.update_observations(pulled_arm, reward)
            self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm]*(self.t - 1) + reward)/self.t
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            