# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 11:33:14 2022

@author: carlo
"""
import numpy as np
from Environment import env
from UCB_Learner import ucb
import matplotlib.pyplot as plt


# build environment
p = [0.5, 0.1, 0.3, 0.9]
pricing_env = env(p)

# build agent (the algo)
ag1 = ucb(len(p))
T = 1000                   # length of the experiment
opt = np.max(p)            # probability of the optimal arm
N_exp = 10                 # number of times I run the experiment

R = []                     # collect cumulative regret curves for all experiments 
for _ in range(N_exp):
    instant_regret = []                             # to store the istantaneous regret
    for t in range(T):
        pulled_arm = ag1.act()                      # pull an arm selected by the algo
        rew = pricing_env.round(pulled_arm)         # simulate an observation from the pulled arm
        ag1.update(pulled_arm, rew)                 # update the agent according to the observation
        instant_regret.append(opt - rew)            # istantaneous regret
    cumulative_regret = np.cumsum(instant_regret)   # curve of the cumulative regret
    ag1.reset()
    R.append(cumulative_regret)

mean_R = np.mean(R, axis=0)                         # mean of the curves of all experiments
std_dev = np.std(R, axis=0)/np.sqrt(N_exp)          # standard dev of the curves of all the experiments, to build IC of the curve  
    
# plot the result
plt.plot(mean_R)
plt.fill_between(range(T), mean_R - std_dev, mean_R + std_dev, alpha=0.4)