import numpy as np

from Classes import *

T = 10
matrix = np.array([[0.,   0.5,   0.5,   0.,    0.],
                     [0.,   0.,    0.5,   0.5,   0.],
                     [0.,   0.,    0.,    0.5,   0.5],
                     [0.5,  0.,    0.,    0.,    0.5],
                     [0.5,  0.5,   0.,    0.,    0.]])
transition_prob_listofmatrix = [matrix for i in range(3)]
vec = 100*np.ones(6)
dir_params_listofvector = [vec for i in range(3)]
pois_param_vector = [200 for i in range(3)]
mat = np.array([[0.1,  0.1,  0.2,  0.3],
                  [0.5,  0.1,  0.0,  0.1],
                  [0.5,  0.1,  0.0,  0.1],
                  [0.5,  0.1,  0.0,  0.1],
                  [0.5,  0.1,  0.0,  0.1]]  )
conversion_rate_listofmatrix = [mat for i in range(3)]
margin_matrix = np.array([[10, 15, 20, 25],
                            [30, 45, 50, 75],
                            [30, 45, 60, 75],
                            [1, 25, 60, 95],
                            [1, 25, 50, 95]])

env = Hyperparameters(transition_prob_listofmatrix,dir_params_listofvector, pois_param_vector, conversion_rate_listofmatrix, margin_matrix)

day = Day(env, np.zeros(5, dtype=int))
day.run_simulation()
print(day.profit)

