import copy

import numpy as np

time_phases = [30, 60]

lam = 0.85
matrix = np.array([[0.00,       0.00,    0.80,        lam*0.80,    0.00],
                   [0.75,       0.00,    lam*0.70,    0.00,        0.00],
                   [lam*0.50,   0.80,    0.00,        0.00,        0.00],
                   [0.00,       0.00,    lam*0.50,    0.00,        0.75],
                   [0.00,       0.70,    lam*0.50,    0.00,        0.00]])

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
low_matS = np.array([[0.20,   0.15,  0.10,  0.05],
                     [0.10,   0.05,  0.00,  0.00],
                     [0.30,   0.22,  0.20,  0.10],
                     [0.29,   0.23,  0.20,  0.15],
                     [0.27,   0.20,  0.25,  0.20]])
low_matCA = np.array([[0.30,   0.30,  0.20,  0.10],
                      [0.30,   0.27,  0.17,  0.10],
                      [0.15,   0.07,  0.01,  0.00],
                      [0.10,   0.02,  0.01,  0.00],
                      [0.15,   0.05,  0.03,  0.00]])
low_matCG = np.array([[0.15,   0.15,  0.10,  0.05],
                      [0.20,   0.17,  0.08,  0.05],
                      [0.23,   0.17,  0.14,  0.10],
                      [0.15,   0.10,  0.05,  0.00],
                      [0.17,   0.30,  0.25,  0.20]])

conversion_rate_listofmatrix.append(copy.deepcopy([low_matS, low_matCA, low_matCG]))
conversion_rate_listofmatrix.append(copy.deepcopy([matS, matCA, matCG]))


margin_matrix = np.array([[190,   200, 240, 330],
                          [80,     90, 140, 170],
                          [20,     45,  50,  60],
                          [45,     65,  75,  84],
                          [30,     45,  57,  65]])

meppp = np.array([[0.1, 1, 3,   1.5, 2],
                  [0.2, 3, 0.5, 0.5, 1],
                  [0.1, 2, 3,   1,   1]], dtype=float)
data_dict = {
    "tr_prob": transition_prob_listofmatrix,
    "dir_par": dir_params_listofvector,
    "pois_par": pois_param_vector,
    "conv_rate": conversion_rate_listofmatrix,
    "margin": margin_matrix,
    "time_phases": time_phases,
    "meppp": meppp
}