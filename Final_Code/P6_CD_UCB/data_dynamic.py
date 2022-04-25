import numpy as np

time_phases = [100, 250]
matrix = np.array([[0.,   0.5,   0.5,   0.,    0.],
                   [0.,   0.,    0.5,   0.5,   0.],
                   [0.,   0.,    0.,    0.5,   0.5],
                   [0.5,  0.,    0.,    0.,    0.5],
                   [0.5,  0.5,   0.,    0.,    0.]])
transition_prob_listofmatrix = [matrix for i in range(3)]
vec = 100*np.ones(6)
dir_params_listofvector = [vec for i in range(3)]
pois_param_vector = [500 for i in range(3)]

conversion_rate_listofmatrix = []
mat = np.array([[0.1,   0.1,  0.2,  0.3],
                [0.5,   0.4,  0.3,  0.1],
                [0.1,   0.4,  0.3,  0.1],
                [0.5,   0.1,  0.1,  0.01],
                [0.01,  0.7,  0.1,  0.3]])
conversion_rate_listofmatrix.append([mat for i in range(3)])
mat = np.array([[0.1,   0.1,  0.2,  0.3],
                [0.1,   0.2,  0.2,  0.3],
                [0.1,   0.4,  0.3,  0.1],
                [0.5,   0.1,  0.1,  0.01],
                [0.01,  0.4,  0.1,  0.3]])
conversion_rate_listofmatrix.append([mat for i in range(3)])
mat = np.array([[0.5,   0.4,  0.3,  0.1],
                [0.2,   0.4,  0.3,  0.1],
                [0.1,   0.4,  0.3,  0.1],
                [0.2,   0.2,  0.1,  0.1],
                [0.4,   0.2,  0.1,  0.3]])
conversion_rate_listofmatrix.append([mat for i in range(3)])

margin_matrix = np.array([[10, 15, 20, 25],
                          [30, 45, 50, 75],
                          [30, 45, 60, 75],
                          [1, 25, 60, 95],
                          [1, 25, 50, 35]])

data_dict = {
    "tr_prob": transition_prob_listofmatrix,
    "dir_par": dir_params_listofvector,
    "pois_par": pois_param_vector,
    "conv_rate": conversion_rate_listofmatrix,
    "margin": margin_matrix,
    "time_phases": time_phases
}