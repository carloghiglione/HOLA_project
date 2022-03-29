# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 15:25:16 2022

@author: carlo
"""
import numpy as np
import matplotlib.pyplot as plt
from Environment import *
from TS_Learner import *
from Greedy_Learner import *


n_arms = 4
p = np.array([0.15, 0.1, 0.1, 0.35])  # probabilities of bernoulli of each arm (candidate price)
opt = p[3]   # the best is the 4-th candidate price, is the Clairvoyant

T = 300  # horizon of experiment

n_experiments = 100   # number of experiemnt I perform
ts_rewards_per_experiment = []   # reward for Thompson Learner
gr_rewards_per_experiment = []   # reward for Greedy Learner


for e in range(0, n_experiments):
    env = Environment(n_arms = n_arms, probabilities = p)
    ts_learner = TS_Learner(n_arms = n_arms)
    gr_learner = GR_Learner(n_arms = n_arms)
    for t in range(0, T):
        # Thomson Sampling Learner
        pulled_arm = ts_learner.pull_arm()  # find the pulled arm by TS
        reward = env.round(pulled_arm)      # observe data of pulled_arm from Environment
        ts_learner.update(pulled_arm, reward)
        
        # Greedy Learner
        pulled_arm = gr_learner.pull_arm()  # find the pulled arm by TS
        reward = env.round(pulled_arm)      # observe data of pulled_arm from Environment
        gr_learner.update(pulled_arm, reward)
        
    ts_rewards_per_experiment.append(ts_learner.collected_rewards)
    gr_rewards_per_experiment.append(gr_learner.collected_rewards)
    
   
plt.figure(0)
plt.xlabel('t')
plt.ylabel('Regret')
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), 'r')   # increases logarithmically
plt.plot(np.cumsum(np.mean(opt - gr_rewards_per_experiment, axis=0)), 'g')   # increases linearly
plt.legend(['TS', 'GR'])

# actually Cumulative_Reward(time) = cumsum_time(Collected_Rewards(time))
# by definition, the Regret(time) = Optimal_Reward - Cumulative_Reward(time)
# since I deal with the Expected Reward and Expected Regret, for each time I perform mean over the 100 experiments
# to find the Expected_Reward(time) and compute the Cumulative_Expected_Reward(time)
# the mean of the Clairvoyant is p = 0.35 (mean of the Bernoulli rand var modeling the optimal candidate price),
# so the Expected_Reward_Clairvoyant(time) = 0.35
# Regret(time) = Expected_Reward_Clairvoyant(time) - Cumulative_Expected_Reward(time)
    
    
    

