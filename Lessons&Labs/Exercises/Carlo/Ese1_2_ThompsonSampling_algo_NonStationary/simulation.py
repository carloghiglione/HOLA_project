# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 15:24:15 2022

@author: carlo
"""
import numpy as np
import matplotlib.pyplot as plt 
from Environment import non_stat_env
from SW_TS import SW_TS

# matrix of probs, arms are columns, phases along row
p = np.array([[0.5, 0.4, 0.3, 0.2],
             [0.4, 0.5, 0.2, 0.3],
             [0.3, 0.2, 0.4, 0.6]])
n_arms = p.shape[1]

T = 1000
env = non_stat_env(n_arms, p, T)

C = 4
window_size = C*int(np.sqrt(T)) # should be order of C*sqrt(horizon) for some C

ag1 = SW_TS(n_arms, window_size)
opt = np.max(p, axis=1)
N_exp = 25

R = []                     # collect cumulative regret curves for all experiments 
for _ in range(N_exp):
    ag1.reset()
    instant_regret = []                             # to store the istantaneous regret
    for t in range(T):
        pulled_arm = ag1.act()                      # pull an arm selected by the algo
        rew = env.round(pulled_arm)                 # simulate an observation from the pulled arm
        ag1.update(pulled_arm, rew)                 # update the agent according to the observation
        phase = env.phase                           # tell current phase
        instant_regret.append(opt[phase] - p[phase, pulled_arm])  # istantaneous regret
    cumulative_regret = np.cumsum(instant_regret)   # curve of the cumulative regret
    R.append(cumulative_regret)

mean_R = np.mean(R, axis=0)                         # mean of the curves of all experiments
std_dev = np.std(R, axis=0)/np.sqrt(N_exp)          # standard dev of the curves of all the experiments, to build IC of the curve  
    
# plot the result
plt.plot(mean_R)
plt.fill_between(range(T), mean_R - std_dev, mean_R + std_dev, alpha=0.4)
    