# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 11:33:14 2022

@author: carlo
"""
import numpy as np
from Environment import env
from UCB_Learner import ucb
# from UCB_Learner_alternative import ucb2
import matplotlib.pyplot as plt

# issues occour when I have two good arms almost equal, should run for long time T
# build environment
p = [0.5, 0.1, 0.2, 0.9]
prices = [100, 400, 300, 60]
pricing_env = env(p, prices)

# build agent (the algo)
ag1 = ucb(len(p), prices)
T = 10000                                            # length of the experiment
opt = np.max([a*b for a,b in zip(p, prices)])       # optimum in the maximum (prob of sell)*(price of sell)
N_exp = 5                                          # number of times I run the experiment

R = []                     # collect cumulative regret curves for all experiments 
for _ in range(N_exp):
    ag1.reset()
    instant_regret = []                             # to store the istantaneous regret
    for t in range(T):
        pulled_arm = ag1.act()                      # pull an arm selected by the algo
        rew = pricing_env.round(pulled_arm)         # simulate an observation from the pulled arm
        ag1.update(pulled_arm, rew)                 # update the agent according to the observation
        instant_regret.append(opt - rew)            # istantaneous regret
    cumulative_regret = np.cumsum(instant_regret)   # curve of the cumulative regret
    R.append(cumulative_regret)

mean_R = np.mean(R, axis=0)                         # mean of the curves of all experiments
std_dev = np.std(R, axis=0)/np.sqrt(N_exp)          # standard dev of the curves of all the experiments, to build IC of the curve  
    
# plot the result
plt.plot(mean_R)
plt.fill_between(range(T), mean_R - std_dev, mean_R + std_dev, alpha=0.4)