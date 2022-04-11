# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 18:06:59 2022

@author: carlo
"""
from scipy.optimize import linear_sum_assignment
from Non_Stationary_Environment import Non_Stationary_Environment
from CUMSUM_UCB_Matching import CUMSUM_UCB_Matching
import numpy as np
import matplotlib.pyplot as plt

# phase 0 adiajency matrix
p0 = np.array([[0.25, 0.25, 0.25],
               [0.5, 0.25, 0.25],
               [0.25, 0.25, 1.0]])
# phase 1 adiajency matrix
p1 = np.array([[1.0, 0.25, 0.25],
               [0.5, 0.25, 0.25],
               [0.25, 0.25, 1.0]])
# phase 2 adiajency matrix
p2 = np.array([[1.0, 0.25, 0.25],
               [0.5, 1.0, 0.25],
               [0.25, 0.25, 1.0]])
P = [p0, p1, p2]                    # collect all matrices
T = 3000                            # time of the experiment
n_exp = 10                          # number of experiments
regret_cumsum = np.zeros((n_exp, T))
# detections = [[] for _ in range(n_exp)]
M = 100
eps = 0.1
h = np.log(T)*2

for j in range(n_exp):
    print(j)
    env_CD = Non_Stationary_Environment(p0.size, P, T)
    learner_CD = CUMSUM_UCB_Matching(p0.size, *p0.shape, M, eps, h)
    opt_rew = []
    rew_CD = []
    for t in range(T):
        p = P[int(t / env_CD.phase_size)]
        opt = linear_sum_assignment(-p)
        opt_rew.append(p[opt].sum())
        
        pulled_arms = learner_CD.pull_arm()
        reward = env_CD.round(pulled_arms)
        learner_CD.update(pulled_arms, reward)
        rew_CD.append(reward.sum())
    regret_cumsum[j,:] = np.cumsum(opt_rew) - np.cumsum(rew_CD)
    
plt.figure(0)
plt.ylabel('Regret')
plt.xlabel('t')
plt.plot(np.mean(regret_cumsum, axis=0))
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    