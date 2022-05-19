import numpy as np

matrix = np.array([[0.,   0.5,   0.5,   0.,    0.],
                   [0.,   0.,    0.5,   0.5,   0.],
                   [0.,   0.,    0.,    0.5,   0.5],
                   [0.5,  0.,    0.,    0.,    0.5],
                   [0.5,  0.5,   0.,    0.,    0.]])
transition_prob_listofmatrix = [matrix for i in range(3)]
vec = 100*np.ones(6)
dir_params_listofvector = [vec for i in range(3)]
pois_param_vector = [1000 for i in range(3)]
mat = np.array([[0.1,   0.2,  0.3,  0.2],
                [0.5,   0.4,  0.3,  0.1],
                [0.4,   0.3,  0.2,  0.1],
                [0.5,   0.1,  0.1,  0.1],
                [0.1,   0.3,  0.1,    0]])

conversion_rate_listofmatrix = [mat for i in range(3)]
margin_matrix = np.array([[10, 15, 20, 25],
                          [30, 35, 40, 45],
                          [20, 25, 30, 35],
                          [1,  5,   7,  9],
                          [5,  8,  11, 13]])

data_dict = {
    "tr_prob": transition_prob_listofmatrix,
    "dir_par": dir_params_listofvector,
    "pois_par": pois_param_vector,
    "conv_rate": conversion_rate_listofmatrix,
    "margin": margin_matrix
}