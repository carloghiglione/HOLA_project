# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 16:40:52 2022

@author: carlo
"""
from Environment import Environment
import numpy as np

# time is divided in phases, from one phase to the other the behaviour of rewards changes
# now probabilities is not anymore a vector of params of bernoulli, one for each arm
# it is a matrix where each row is an arm and each col tells the value of bernoulli param in each phase

class Non_Stationary_Environment(Environment):
    def __init__(self, n_arms, probabilities, horizon):
        super().__init__(n_arms, probabilities)
        self.t = 0                              # initial time
        n_phases = len(self.probabilities)      # number of phases
        self.phases_size = horizon/n_phases     # size of a phase
        
    def round(self, pulled_arm):
        current_phase = int(self.t / self.phases_size)      # current phase
        p = self.probabilities[current_phase][pulled_arm]  # Bernoulli param at current phase for pullee_arm
        reward = np.random.binomial(1,p)                   # generate the current observed reward
        self.t += 1
        return reward