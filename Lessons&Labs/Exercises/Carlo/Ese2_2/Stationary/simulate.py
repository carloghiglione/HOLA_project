# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 16:36:22 2022

@author: carlo
"""
import numpy as np
from UCB_matching import UCB_Matching
from Environment import environment
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# assignment matrix, probs[i,j] is the probability Buyer i buys an item from Seller j
p = np.array([[0.25, 1.0, 0.25],
              [0.5, 0.25, 0.25],
              [0.25, 0.25, 1.0]])

# clairvoyant solution, solution knowing all nodes a priori
opt = linear_sum_assignment(-p)

n_exp = 10
T = 3000
regret_ucb = np.zeros((n_exp, T))

for e in range(n_exp):
    learner = UCB_Matching(p.size, 3,3)
    print(e)
    rew_UCB = []
    opt_rew = []
    env = environment(p)
    for t in range(T):
        pulled_arms = learner.pull_arm()
        rewards = env.round(pulled_arms)
        learner.update(pulled_arms, rewards)
        rew_UCB.append(rewards.sum())
        opt_rew.append(p[opt].sum())
    regret_ucb[e, :] = np.cumsum(opt_rew) - np.cumsum(rew_UCB)
    
plt.figure(0)
plt.plot(regret_ucb.mean(axis=0))
plt.ylabel('Regret')
plt.xlabel('t')
        
        
        
        
        
        
        
        
        
        
        
        
        