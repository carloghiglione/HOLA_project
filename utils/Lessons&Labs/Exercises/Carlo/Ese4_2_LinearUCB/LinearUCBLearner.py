# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 15:36:23 2022

@author: carlo
"""

import numpy as np

class LinearUCBLearner():
    
    def __init__(self, arms_features):       # input: matrix of arms features vectors defined in the environment
        self.arms = arms_features            # takes the arms' features vectors
        self.dim = arms_features.shape[1]    # dimension of feature vector
        self.collected_rewards = []          # store rewards
        self.pulled_arms = []                # store pulled arms
        self.c = 2.0                         # exploration factor
        self.M = np.identity(self.dim)       # matrix M
        self.b = np.atleast_2d(np.zeros(self.dim)).T        # vector b of dimesion dim, has shape (dim, 1) thanks to atleast2d function, need it to compute dot profucts
        self.theta = np.dot(np.linalg.inv(self.M), self.b)  # vector of params theta
        
    # function to compute values of Upper Confidence Bounds
    def compute_ucbs(self): 
        self.theta = np.dot(np.linalg.inv(self.M), self.b)  # update values of theta
        ucbs = []                                           # list of ucbs
        for arm in self.arms:                               # iterate over all arms
            arm = np.atleast_2d(arm).T                      # reshape vector arm of size (dim, 1)
            ucb = np.dot(self.theta.T, arm) + self.c * np.sqrt(np.dot(arm.T, np.dot(np.linalg.inv(self.M), arm)))  # compute Upper Confidence Bound associated to current arm
            ucbs.append(ucb[0][0])                          # add the value to the list
        return ucbs
    
    # function to pull the arm with maximum UCB
    def pull_arm(self):
        ucbs = self.compute_ucbs()
        return np.argmax(ucbs)
    
    # function to update matrix M and vector b according to the arm we pulled and the reward drawn in the last round
    def update_estimation(self, arm_idx, reward):
        arm = np.atleast_2d(self.arms[arm_idx]).T
        self.M += np.dot(arm, arm.T)   # matrix M is updated summing to the current M the dot product of feature vector of the arm we pulled with itself
        self.b += reward * arm         # vector b is updated summing to the current b the product of the feature vector with the last reward
     
    # fucntion to update according to the pulled arm and the collected reward
    def update(self, arm_idx, reward):
        self.pulled_arms.append(arm_idx)
        self.collected_rewards.append(reward)
        self.update_estimation(arm_idx, reward)
    
    
    
    
    
    
    
    
    
    