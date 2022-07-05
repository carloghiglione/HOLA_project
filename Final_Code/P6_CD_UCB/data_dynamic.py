import copy

import numpy as np

time_phases = [30, 60, 90]

lam = 0.85
matrix = np.array([[0.,         0.,     0.80,       lam*0.80,   0.],
                   [0.75,       0.,     lam*0.70,   0.,         0.],
                   [lam*0.50,   0.80,   0.,         0.,         0.],
                   [0.,         0.,     lam*0.50,   0.,         0.75],
                   [0.,         0.70,   lam*0.5,    0.,         0.]])


transition_prob_listofmatrix = [matrix for i in range(3)]
vecS = 10000*np.array([20,  5, 10, 25, 30, 15], dtype=float)
vecCA = 10000*np.array([15, 30, 25, 10,  5, 15], dtype=float)
vecCG = 10000*np.array([15,  5,  5, 20, 30, 20], dtype=float)
dir_params_listofvector = [vecS, vecCA, vecCG]

pois_param_vector = 10*np.array([250, 450, 300])

conversion_rate_listofmatrix = []

matS = np.array([[0.05,   0.01,  0.00,  0.00],
                 [0.20,   0.10,  0.05,  0.00],
                 [0.60,   0.55,  0.50,  0.45],
                 [0.30,   0.25,  0.20,  0.10],
                 [0.27,   0.20,  0.25,  0.20]])
matCA = np.array([[0.23,   0.30,  0.27,  0.25],
                  [0.30,   0.37,  0.35,  0.30],
                  [0.03,   0.07,  0.01,  0.00],
                  [0.00,   0.05,  0.03,  0.00],
                  [0.15,   0.05,  0.03,  0.0]])
matCG = np.array([[0.15,   0.07,  0.05,  0.00],
                  [0.38,   0.35,  0.30,  0.10],
                  [0.30,   0.30,  0.27,  0.20],
                  [0.23,   0.20,  0.17,  0.10],
                  [0.17,   0.30,  0.25,  0.20]])

conversion_rate_listofmatrix.append(copy.deepcopy([matS, matCA, matCG]))
low_matS = np.array([[0.20,   0.10,  0.05,  0.00],
                     [0.05,   0.01,  0.00,  0.00],
                     [0.30,   0.25,  0.20,  0.10],
                     [0.60,   0.55,  0.50,  0.45],
                     [0.27,   0.20,  0.25,  0.20]])
low_matCA = np.array([[0.23,   0.30,  0.27,  0.25],
                      [0.30,   0.37,  0.35,  0.30],
                      [0.03,   0.07,  0.01,  0.00],
                      [0.00,   0.05,  0.03,  0.00],
                      [0.15,   0.05,  0.03,  0.0]])
low_matCG = np.array([[0.15,   0.07,  0.05,  0.00],
                      [0.38,   0.35,  0.30,  0.10],
                      [0.30,   0.30,  0.27,  0.20],
                      [0.23,   0.20,  0.17,  0.10],
                      [0.17,   0.30,  0.25,  0.20]])
conversion_rate_listofmatrix.append(copy.deepcopy([low_matS, low_matCA, low_matCG]))
conversion_rate_listofmatrix.append(copy.deepcopy([matS, matCA, matCG]))

# 1: 200 all
# 2: 100 all,200 all
# 3: no
# 4: 200, all but 3
# 5: 100 2, 200 1 2
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