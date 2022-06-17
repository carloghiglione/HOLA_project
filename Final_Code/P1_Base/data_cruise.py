import numpy as np

#order [capitano, cena, cocktail, sport, massaggi]
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
pois_param_vector = 10*np.array([100, 600, 300])
matS = np.array([[0.10,   0.07,  0.05,  0.01],
                 [0.15,   0.10,  0.10,  0.05],
                 [0.40,   0.37,  0.35,  0.33],
                 [0.25,   0.25,  0.20,  0.15],
                 [0.15,   0.13,  0.10,  0.03]])
matCA = np.array([[0.27,   0.25,  0.20,  0.15],
                  [0.40,   0.35,  0.35,  0.30],
                  [0.15,   0.13,  0.10,  0.05],
                  [0.15,   0.10,  0.10,  0.05],
                  [0.15,   0.13,  0.10,  0.03]])
matCG = np.array([[0.10,   0.07,  0.05,  0.03],
                  [0.25,   0.20,  0.15,  0.10],
                  [0.40,   0.37,  0.35,  0.30],
                  [0.23,   0.20,  0.17,  0.10],
                  [0.15,   0.13,  0.10,  0.03]])



matS = np.array([[0.10,   0.07,  0.05,  0.01],
                 [0.15,   0.10,  0.10,  0.05],
                 [0.50,   0.40,  0.35,  0.33],
                 [0.30,   0.25,  0.20,  0.10],
                 [0.20,   0.15,  0.10,  0.05]])
matCA = np.array([[0.26,   0.25,  0.20,  0.15],
                  [0.40,   0.35,  0.35,  0.30],
                  [0.13,   0.10,  0.07,  0.01],
                  [0.15,   0.10,  0.10,  0.05],
                  [0.15,   0.13,  0.10,  0.03]])
matCG = np.array([[0.15,   0.10,  0.07,  0.03],
                  [0.38,   0.35,  0.30,  0.25],
                  [0.40,   0.35,  0.30,  0.25],
                  [0.23,   0.20,  0.17,  0.10],
                  [0.15,   0.13,  0.10,  0.03]])

conversion_rate_listofmatrix = [matS, matCA, matCG]

margin_matrix = np.array([[170,   230, 250, 300],
                          [80,    100, 130, 150],
                          [20,     30,  50,  60],
                          [45,     65,  70,  84],
                          [25,     45,  50,  60]])


meppp = np.array([[0.1, 3, 7, 3, 4],
                  [0.1, 3, 7, 3, 4],
                  [0.1, 3, 7, 3, 4]], dtype=float)
data_dict = {
    "tr_prob": transition_prob_listofmatrix,
    "dir_par": dir_params_listofvector,
    "pois_par": pois_param_vector,
    "conv_rate": conversion_rate_listofmatrix,
    "margin": margin_matrix,
    "meppp": meppp
}
