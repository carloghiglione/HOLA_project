import numpy as np

lam = 0.85
matrix = np.array([[0.,         0.,     0.80,       lam*0.80,   0.],
                   [0.75,       0.,     lam*0.70,   0.,         0.],
                   [lam*0.50,   0.80,   0.,         0.,         0.],
                   [0.,         0.,     lam*0.50,   0.,         0.75],
                   [0.,         0.70,   lam*0.5,    0.,         0.]])

transition_prob_listofmatrix = [matrix for i in range(3)]
vecS  = 10000*np.array([20,  5, 10, 25, 30, 15], dtype=float)
vecCA = 10000*np.array([15, 30, 25, 10,  5, 15], dtype=float)
vecCG = 10000*np.array([15,  5,  5, 20, 30, 20], dtype=float)
dir_params_listofvector = [vecS, vecCA, vecCG]
#pois_param_vector = 10*np.array([100, 600, 300])
pois_param_vector = [[100,  150],
                     [450,  300]]
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


conversion_rate_listofmatrix = [matS, matCA, matCG]

margin_matrix = np.array([[170,   230, 250, 300],
                          [80,     90, 130, 170],
                          [15,     30,  45,  60],
                          [45,     65,  75,  84],
                          [30,     45,  55,  65]])

meppp = np.array([[0.1, 1, 3,   1.5, 2],
                  [0.2, 3, 0.5, 0.5, 1],
                  [0.1, 2, 3,   1,   1]], dtype=float)

feat_ass = np.array([[0, 0],
                     [1, 2]], dtype=int)


data_dict = {
    "tr_prob": transition_prob_listofmatrix,
    "dir_par": dir_params_listofvector,
    "pois_par": pois_param_vector,
    "conv_rate": conversion_rate_listofmatrix,
    "margin": margin_matrix,
    "feat_ass": feat_ass,
    "meppp": meppp
}