# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 15:09:06 2022

@author: carlo
"""
import numpy as np
class learner:
    
    def __init__(self, n_arms):
        self.t = 0
        self.n_arms = n_arms
        self.rewards = []    # to collect all rewards
        self.reward_per_arm = [[] for _ in range(n_arms)]    # list of lists to collect rewards for each arm
        self.pulled = []
        
    # # function to reset to initial condition
    # def reset(self):
    #     self.__init__(self.n_arms, self.prices, self.T)
    
    # perform one step of algo to select the arm that I pull, specific of the algorithm I use
    def act(self):
        pass
    
    # collect output of algo and append the reward
    def update_observations(self, arm_pulled, reward):
        self.t += 1                                      # update time
        self.rewards.append(reward)
        self.reward_per_arm[arm_pulled].append(reward)
        self.pulled.append(arm_pulled)