import numpy as np

matrix = np.array([[0.,   0.5,   0.5,   0.,    0.],
                   [0.,   0.,    0.5,   0.5,   0.],
                   [0.,   0.,    0.,    0.5,   0.5],
                   [0.5,  0.,    0.,    0.,    0.5],
                   [0.5,  0.5,   0.,    0.,    0.]])
transition_prob_listofmatrix = [matrix for i in range(3)]
vec = np.ones(6)
dir_hyperparams_listofvector = [vec for i in range(3)]
dir_pois_vector = 100*np.ones(3)
pois_param_vector = 500*np.ones(3)
mat = np.array([[0.1,   0.1,  0.2,  0.3],
                [0.5,   0.4,  0.3,  0.1],
                [0.1,   0.4,  0.3,  0.1],
                [0.5,   0.1,  0.1,  0.01],
                [0.01,  0.7,  0.1,  0.3]])
conversion_rate_listofmatrix = [mat for i in range(3)]
margin_matrix = np.array([[10, 15, 20, 25],
                          [30, 45, 50, 75],
                          [30, 45, 60, 75],
                          [1, 25, 60, 95],
                          [1, 25, 50, 35]])

data_dict = {
    "tr_prob": transition_prob_listofmatrix,
    "dir_hpar": dir_hyperparams_listofvector,
    "dir_poi": dir_pois_vector,
    "pois_poi": pois_param_vector,
    "conv_rate": conversion_rate_listofmatrix,
    "margin": margin_matrix
}