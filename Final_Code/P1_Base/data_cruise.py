import numpy as np

#order [capitano, cena, cocktail, sport, massaggi]
lam = 0.75
matrix = 0.5*np.array([[0.,         0.,     0.80,       lam*0.80,   0.],
                   [0.75,       0.,     lam*0.70,   0.,         0.],
                   [lam*0.50,   0.80,   0.,         0.,         0.],
                   [0.,         0.,     lam*0.50,   0.,         0.75],
                   [0.,         0.70,   lam*0.5,    0.,         0.]])
transition_prob_listofmatrix = [matrix for i in range(3)]
vecS = 100*np.array([0.20, 0.05, 0.10, 0.25, 0.30, 0.15], dtype=float)
vecCA = 100*np.array([0.15, 0.30, 0.25, 0.10, 0.05, 0.15], dtype=float)
vecCG = 100*np.array([0.15, 0.05, 0.05, 0.20, 0.30, 0.20], dtype=float)
dir_params_listofvector = [vecS, vecCA, vecCG]
pois_param_vector = [100, 600, 300]
matS = np.array([[0.02, 0.0015,  0.01, 0.005],
                 [0.15,   0.08,  0.10,  0.08],
                 [0.30,   0.27,  0.25,  0.23],
                 [0.25,   0.23,  0.20,  0.18],
                 [0.15,   0.13,  0.10,  0.09]])
matCA = np.array([[0.10,   0.08,  0.07,  0.06],
                  [0.30,   0.28,  0.26,  0.24],
                  [0.12,   0.11,  0.10,  0.08],
                  [0.06,   0.055, 0.05,  0.03],
                  [0.15,   0.13,  0.10,  0.09]])
matCG = np.array([[0.05,   0.04,  0.03,  0.02],
                  [0.20,   0.18,  0.15,  0.13],
                  [0.25,   0.20,  0.18,  0.15],
                  [0.23,   0.20,  0.17,  0.14],
                  [0.15,   0.13,  0.10,  0.09]])

conversion_rate_listofmatrix = [matS, matCA, matCG]
margin_matrix = np.array([[250*0.75, 250*0.9, 250, 250*1.1],
                          [100*0.75, 100*0.9, 100, 100*1.1],
                          [30*0.75,   30*0.9,  30,  30*1.1],
                          [70*0.75,   70*0.9,  70,  70*1.1],
                          [50*0.75,   50*0.9,  50,  50*1.1]])
meppp = np.array([[0.05, 2.5, 6, 3, 4],
                  [0.05, 2.5, 6, 3, 4],
                  [0.05, 2.5, 6, 3, 4]], dtype=float)
data_dict = {
    "tr_prob": transition_prob_listofmatrix,
    "dir_par": dir_params_listofvector,
    "pois_par": pois_param_vector,
    "conv_rate": conversion_rate_listofmatrix,
    "margin": margin_matrix,
    "meppp": meppp
}