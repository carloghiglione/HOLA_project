# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 14:47:44 2022

@author: carlo
"""
import numpy as np

# class that generates the data
class Environment():
    def __init__(self, n_arms, probabilities):
        self.n_arms = n_arms                   # number of arms (price candidates)
        self.probabilities = probabilities     # real probability distr of each arm
    
    # to simulate an observation from the real distribution of pulled_arm
    def round(self, pulled_arm):
        reward = np.random.binomial(1, self.probabilities[pulled_arm])
        return reward
    
# this is a case where the reward curve can assume only {0,1} values
# so the random variable modeling the reward curve for each proposed price is
# reward(price) ~ Bernoulli(p(price))  p in [0,1]
# as a consequence, the Environment reward (the observed data) is produced as Beronoulli distribution 
    