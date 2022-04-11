# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 17:28:52 2022

@author: carlo
"""
from UCB_matching import UCB_Matching
import numpy as np
from scipy.optimize import linear_sum_assignment
from Cumsum import CUMSUM


class CUMSUM_UCB_Matching(UCB_Matching):
    def __init__(self, n_arms, n_rows, n_cols, M=100, eps=0.05, h=20, alpha=0.01):
        super().__init__(n_arms, n_rows, n_cols)
        self.change_detection = [CUMSUM(M, eps, h) for _ in range(n_arms)]   # initialize CUMSUM change detection algo for each arm
        self.valid_rewards_per_arms = [[] for _ in range(n_arms)]            # keep track of valid rewards for each arm
        self.detections = [[] for _ in range(n_arms)]                        # for each arm collect times when detection was flagged
        self.alpha = alpha
        
    def pull_arm(self):
        if np.random.binomial(1, 1-self.alpha):                              # with prob 1-alpha we select the super arm wich maximizes the sum of upper confidence bounds
            upper_conf = self.empirical_means + self.confidence
            upper_conf[np.isinf(upper_conf)] = 1e3
            row_ind, col_ind = linear_sum_assignment(-upper_conf.reshape(self.n_rows, self.n_cols))
            return row_ind, col_ind
        else:                                                                # otherwise we create a random matrix and then solve optimization problem over it to generate a random matching
            costs_random = np.random.randint(0, 10, size=(self.n_rows, self.n_cols))
            return linear_sum_assignment(costs_random)
        
    def update(self, pulled_arms, rewards):
        self.t += 1
        pulled_arm_flat = np.ravel_multi_index(pulled_arms, (self.n_rows, self.n_cols))
        for pulled_arm, reward in zip(pulled_arm_flat, rewards):    # for each arm we pull, we ask change detection algo for that specific arm
            if self.change_detection[pulled_arm].update(reward):    # if a detection is flagged
                self.detections[pulled_arm].append(self.t)          # add the detected change time
                self.valid_rewards_per_arms[pulled_arm] = []        # reinitialize the valid reward (rewardand samples are synonim) for that arm being empty
                self.change_detection[pulled_arm].reset()           # reset the change detection algo
            self.update_observations(pulled_arm, reward)
            self.empirical_means[pulled_arm] = np.mean(self.valid_rewards_per_arms[pulled_arm])  # compute the empirical mean of the pulled arm only over the valid samples
        total_valid_samples = sum([len(x) for x in self.valid_rewards_per_arms])    # sum of valid samples for each arm
        for a in range(self.n_arms):
            n_samples = len(self.valid_rewards_per_arms[a])
            self.confidence[a] = (2*np.log(total_valid_samples)/n_samples)**0.5 if n_samples > 0 else np.inf  # update confidences for each arm with the correspondig valid samples
    
    
    def update_observations(self, pulled_arm, reward):      # I need to overwrite it since I require to collecy also the valid rewards for each arm
        self.reward_per_arm[pulled_arm].append(reward)
        self.valid_rewards_per_arms[pulled_arm].append(reward)      
        self.rewards = np.append(self.rewards, reward)
                
                
                
                
                
                
                
                
                
                
                
                
                
                
        