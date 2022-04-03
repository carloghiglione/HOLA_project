# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 15:12:06 2022

@author: carlo
"""

import numpy as np

class LinearMabEnvironment():
    
    def __init__(self, n_arms, dim):                                         # input: number of arms, feature vector size
        self.theta = np.random.dirichlet(np.ones(dim), size=1)               # initialize param theta of the bernoulli distr we use to draw the reward, their sum must be one
        self.arms_features = np.random.binomial(1, 0.5, size=(n_arms, dim))  # generate n_arms feature vectors, rows correspond to arms, columns correspond to feature values of each arm
        self.p = np.zeros(n_arms)                                            # probabilities related to each arm
        for i in range(n_arms):
            self.p[i] = np.dot(self.theta, self.arms_features[i])            # compute the probabilities
        
    def round(self, pulled_arm):
        return 1 if np.random.random() < self.p[pulled_arm] else 0           # returns value from Bernoulli of succes probability equal to the arm probability

    def opt(self):                                                           # returns the maximum value of clairvoyant 
        return np.max(self.p)