# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 09:52:43 2022

@author: carlo
"""

import numpy as np
from copy import copy

# function to simulate Diffusion Episode
def simulate_episode(init_prob_matrix, n_step_max):   # inputs: probability matrix and maximum number of time steps we simulate for each episode
    prob_matrix = init_prob_matrix.copy()
    n_nodes = prob_matrix.shape[0]                    # number of nodes in the graph
    initial_active_nodes = np.random.binomial(1, 0.1, size=(n_nodes))   # we define the active nodes at step zero (the"seeds")
    history = np.array([initial_active_nodes])        # dataset that collects the active nodes at each time step in the graph
    active_nodes = initial_active_nodes               # collects all activated nodes up to now
    newly_active_nodes = active_nodes                 # collects nodes activated in the current time step
    t = 0
    while (t < n_step_max) and (np.sum(newly_active_nodes) > 0):  # iterate over the time steps, finishes when we don't have any more node to be activated or I reach the limit of steps
        p = (prob_matrix.T * active_nodes).T          # selects from the probability matrix only the rows related to the active nodes
        activated_edges = p > np.random.rand(p.shape[0], p.shape[1])    # compute the values of the activated edges, each edge is activated if it is bigger than a random number sampled uniformly in [0,1]
        prob_matrix = prob_matrix * ( (p != 0) == activated_edges)      # remove from probability matrix the values related to the previously activated edges
        newly_active_nodes = (np.sum(activated_edges, axis=0) > 0) * (1 - active_nodes)  # compute the values of the newly activated nodes, 
                                                                                         # I sum values of activated edges going to each node, if the sum bigger than one (i.e. at least one edge going to the node is activated), activate the node
                                                                                         # multiply by (1 - active nodes) to impose that newly active nodes can't be previously activated nodes
        active_nodes = np.array(active_nodes + newly_active_nodes)                       # add newly activated nodes to the activated nodes
        history = np.concatenate((history, [newly_active_nodes]), axis=0)                # add the newly activated nodes to the history of the activatation of the graph
        t += 1    # increase time step
    return history


# Credit Assignment approach to estimate the probabilites of the edges directed to a target node of the graph:
# at each episode where target node has been active, assign the credits to the nodes going into the target node
# depending on if these nodes has been active at previous time step
# be k target node, the final estimation of edge probability (i->k) is computed dividing the sum of credits assigned to (i->k)
# by the number of episodes in which node i has been active previously than node k (or i has been active and node k not active)
    
def estimate_probabilities(dataset, node_index, n_nodes):   # input: dataset of diffusion episodes, the target node index, number of nodes of the graph
    estimated_prob = np.ones(n_nodes)*1.0/(n_nodes - 1)     # initialize estimated probabilities of all the nodes to this reference value
    node_credits = np.zeros(n_nodes)                        # initialize variable where I store the credits assigned to each node in all the episodes
    occour_v_active = np.zeros(n_nodes)                     # initialize variable where I store the occurences of each node in all the episodes
    n_episodes = len(dataset)                               # number of episodes
    for episode in dataset:                                 # iterate for each episode of the dataset
        idx_w_active = np.argwhere(episode[:,node_index] == 1).reshape(-1)   # find in which time steps (row) the target node has been activated
        if (len(idx_w_active) > 0) and (idx_w_active > 0):                   
            active_nodes_in_prev_step = episode[idx_w_active - 1,:].reshape(-1)           # check which were the nodes active in the previous step
            node_credits += active_nodes_in_prev_step/np.sum(active_nodes_in_prev_step)   # assign the credits uniformly to all nodes that has been active in the previous step
        for v in range(0, n_nodes):             # iterate over all the nodes
            if v != node_index:                 # for the nodes that are not the target node
                idx_v_active = np.argwhere(episode[:,v] == 1).reshape(-1)                 # check the occurences of each node in each episode (which time steps node v has been active)
                if (len(idx_v_active) > 0) and (idx_v_active < idx_w_active or len(idx_w_active) == 0):   # if node b has been activated at least once in this episode and if the time step of the activation of v is lower than the activation of node w (if w has been activated)
                    occour_v_active[v] += 1     # increase by one the value of v occurences
    estimated_prob = node_credits/occour_v_active     # the estimated probabilities to reach the node is the credits divided by the occourrences of each node in the epispdes
    estimated_prob = np.nan_to_num(estimated_prob)    # remove the nans from the estimated probabilities
    return estimated_prob























