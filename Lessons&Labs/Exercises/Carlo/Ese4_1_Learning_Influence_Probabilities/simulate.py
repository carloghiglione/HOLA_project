# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 10:58:55 2022

@author: carlo
"""

from learning_probabilities import simulate_episode, estimate_probabilities
import numpy as np


# Scenario:
# set of nodes representing users, set of directed edges representing connections among users, each edge has an influence probability
# if node A active and edge(A->B) = 0.2, node B will be active in next step with prob 0.2

# we simulate Diffusion Episodes: set initial active nodes ("seeds") and propagate the activation according to inflence probabilities

# Goal: observing which nodes has been activated at each time step of one episode, we want to estimate the values of the influence probabilities

# MonteCarlo approach: I simulate many episodes randomly setting different seeds and use this dataset to estimate the edges


# set number of nodes
n_nodes = 5
# set number of episodes
n_episodes = 5000
# initialize probability matrix of the edges drawing Unif(0, 0.1)
prob_matrix = np.random.uniform(0, 0.1, (n_nodes, n_nodes))
# node target
node_index = 4

# initialize the dataset of the episodes
dataset = []

# generate all the episodes, build the dataset
for e in range(n_episodes):
    dataset.append(simulate_episode(prob_matrix, n_step_max=10))
    
# compute the estimated probabilities
estimated_probs = estimate_probabilities(dataset, node_index, n_nodes)

print('True P matrix: ', prob_matrix[:,4])
print('Estimated P Matrix: ', estimated_probs)