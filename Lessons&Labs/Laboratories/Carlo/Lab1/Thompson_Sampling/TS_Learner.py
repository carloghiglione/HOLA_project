# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 15:01:49 2022

@author: carlo
"""
from Learner import *

# Thompson Sampling Algo
class TS_Learner(Learner):
    
    def __init__(self, n_arms):
        super().__init__(n_arms)  # usa il costruttore di learner (super-class)
        self.beta_parameters = np.ones((n_arms, 2))  # variable to store parameters of beta distributions for each arm
        
    # to select the arm to  pull 
    # sample from betas of all the arms and find arm that sampled the maximum value
    def pull_arm(self):
        idx = np.argmax(np.random.beta(self.beta_parameters[:,0], self.beta_parameters[:,1])) 
        return idx
    
    def update(self, pulled_arm, reward):
        self.t += 1  # update time
        self.update_observations(pulled_arm, reward)
        self.beta_parameters[pulled_arm,0] = self.beta_parameters[pulled_arm,0] + reward        # update first param of beta
        self.beta_parameters[pulled_arm,1] = self.beta_parameters[pulled_arm,1] + 1.0 - reward  # update second param of beta
        