# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 15:00:56 2022

@author: carlo
"""
# Thomson sampling sliding window
import numpy as np
from Learner import learner

class SW_TS(learner):
    
    def __init__(self, n_arms, window_size):
        super().__init__(n_arms)
        self.n_arms = n_arms
        self.window_size = window_size       # size of the window 
        self.alphas = np.ones(self.n_arms)   # alpha coeff for each arm
        self.betas = np.ones(self.n_arms)    # beta coeff for each arm
        self.t = 0
        
    def reset(self):
        self.__init__(self.n_arms, self.window_size)
        
    def update(self, pulled_arm, reward):
        super().update(pulled_arm, reward)
        for arm_idx in range(self.n_arms):
            n_samples = np.sum(np.array(self.pulled[-self.window_size:])==arm_idx)  # how many times we pulled the current arm in the window
            if n_samples == 0:
                n_sold = 0
            else:
                n_sold = np.sum(self.reward_per_arm[arm_idx][-n_samples:])    # how many times the bernoulli trial was positive in the last n_samples of current arm
            self.alphas[arm_idx] = n_sold + 1                                 # update alpha
            self.betas[arm_idx] = n_samples - n_sold + 1                      # update beta
     
    # choice of the arm
    def act(self):
        samples = [np.random.beta(a=self.alphas[i], b=self.betas[i]) for i in range(self.n_arms)]   # draw sample for each arm
        return np.argmax(samples)