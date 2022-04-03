# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 16:02:10 2022

@author: carlo
"""

import numpy as np
from LinearMabEnvironment import LinearMabEnvironment
from LinearUCBLearner import LinearUCBLearner
import matplotlib.pyplot as plt


# number of arms
n_arms = 10
# time horizon
T = 1000
# number of experiments
n_experiments = 100
# list to collect the rewards
lin_ucb_rewards_per_experiment = []
# dimension of feature vector
fv_dim = 10


# initialize the environment
env = LinearMabEnvironment(n_arms, fv_dim)

for e in range(n_experiments):
    print(f'Experiment number: {e}')
    # build Linear UCB Learner
    lin_ucb_learner = LinearUCBLearner(arms_features=env.arms_features)
    # iterate over the rounds
    for t in range(T):
        # choose the arm to pull
        pulled_arm = lin_ucb_learner.pull_arm()
        # environemnt returns a reward depending on the pulled arm
        reward = env.round(pulled_arm)
        # update the model
        lin_ucb_learner.update(pulled_arm, reward)
    # store the rewards of the current experiment
    lin_ucb_rewards_per_experiment.append(lin_ucb_learner.collected_rewards)
    
# compute the optimum according to the clairvoyant algorithm
opt = env.opt()

# plot cumulative regret, i.e. cumulative sum of the difference between the optimum and the collected rewards in the experiemnts
plt.figure(0)
plt.ylabel('Regret')
plt.xlabel('t')
plt.plot(np.cumsum(np.mean(opt - lin_ucb_rewards_per_experiment, axis=0)), 'r')
plt.legend(['Linear UCB'])














