# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 15:55:29 2022

@author: carlo
"""
# I show how matching algo for bipartite graph works
# I use library already available

import numpy as np
from scipy.optimize import linear_sum_assignment

# assignment matrix, probs[i,j] is the probability Buyer i buys an item from Seller j
probs = np.array([[0.25, 1.0, 0.25],
                  [0.5, 0.25, 0.25],
                  [0.25, 0.25, 1.0]])

# the library works with costs that tries to minimize, so I use as costs - the probabilities
# in this way I find the matching that maximizes the sum of probabilities (weighted matching problem)
costs = -probs

# return rows and cols of the matching
row, col = linear_sum_assignment(costs)
sum_matching_probability = np.sum(probs[row, col])

print(f"Rows: {row}")
print(f"Columns: {col}")
print(f"Optimal Matching sum of probability: {sum_matching_probability}")
