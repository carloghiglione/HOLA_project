# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 17:05:43 2022

@author: carlo
"""
import numpy as np
import matplotlib.pyplot as plt
from Non_Stationary_Environment import Non_Stationary_Environment
from TS_Learner import TS_Learner
from SWTS_Learner import SWTS_Learner


n_arms = 4      # number of candidate prices
n_phases = 4    # number of changes of arms reward behaviors

# matrix that encodes the p params of the bernoulli describing the 4 reward behaviours across the 4 phases
p = np.array([[0.15, 0.1, 0.2, 0.35],
              [0.45, 0.21, 0.2, 0.35],
              [0.1, 0.1, 0.5, 0.15],
              [0.1, 0.21, 0.1, 0.15]])
T = 500                       # time horizon
phases_len = int(T/n_phases)  # phases length
n_experiments = 100           # number of experiments 
ts_rewards_per_experiment = []
swts_rewards_per_experiment = []
window_size = int(T**0.5)     # Sliding Window Size, use the rule to set it proportional to sqrt(Time_horizon)


if __name__ == '__main__':
    
    for e in range(n_experiments):
        ts_env = Non_Stationary_Environment(n_arms, probabilities=p, horizon=T)
        ts_learner = TS_Learner(n_arms)
        
        swts_env = Non_Stationary_Environment(n_arms, probabilities=p, horizon=T)
        swts_learner = SWTS_Learner(n_arms, window_size)
        
        for t in range(T):
            pulled_arm = ts_learner.pull_arm()
            reward = ts_env.round(pulled_arm)
            ts_learner.update(pulled_arm, reward)
            
            pulled_arm = swts_learner.pull_arm()
            reward = swts_env.round(pulled_arm)
            swts_learner.update(pulled_arm, reward)
            
        ts_rewards_per_experiment.append(ts_learner.collected_rewards)
        swts_rewards_per_experiment.append(swts_learner.collected_rewards)
        
    ts_instantaneous_regret = np.zeros(T)
    swts_instantaneous_regret = np.zeros(T)
    opt_per_phases = p.max(axis=1)           # find optimal arm for each phase
    optimum_per_round = np.zeros(T)
       
    for i in range(n_phases):
        t_index = range(i*phases_len, (i+1)*phases_len)  # time indexes of the current phase
        optimum_per_round[t_index] = opt_per_phases[i]   # optimum in the current phase
        ts_instantaneous_regret[t_index] = opt_per_phases[i] - np.mean(ts_rewards_per_experiment, axis=0)[t_index]      # compute cumulative regret
        swts_instantaneous_regret[t_index] = opt_per_phases[i] - np.mean(swts_rewards_per_experiment, axis=0)[t_index]  # compute cumulative regret
        
    # I plot the Expected Reward    
    plt.figure(0)
    plt.plot(np.mean(ts_rewards_per_experiment, axis=0), 'r')
    plt.plot(np.mean(swts_rewards_per_experiment, axis=0), 'b')
    plt.plot(optimum_per_round, 'k--')
    plt.legend(['TS', 'SW-TS', 'Optimum'])
    plt.ylabel('Expected Reward')
    plt.xlabel('t')
    
    # I plot the Cumulative Regret
    plt.figure(1)
    plt.plot(np.cumsum(ts_instantaneous_regret), 'r')
    plt.plot(np.cumsum(swts_instantaneous_regret), 'b')
    plt.legend(['TS', 'SW-TS'])
    plt.ylabel('Regret')
    plt.xlabel('t')
        
        




