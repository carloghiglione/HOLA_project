# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 14:54:25 2022

@author: carlo
"""
import numpy as np

class Learner:
    
    def __init__(self, n_arms):
        self.n_arms = n_arms       # number of arms
        self.t = 0                 # current time (initialized to 0)
        self.rewards_per_arm = x = [[] for i in range(n_arms)]  # list of lists, collects the collected rewards for each round and for each arm
        self.collected_rewards = np.array([])   # to store the values of the rewards at each round
        
    def update_observations(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward)   # add reward to the corresponding list of pulled_arm
        self.collected_rewards = np.append(self.collected_rewards, reward) # add collected reward to numpy list of reward
        
    