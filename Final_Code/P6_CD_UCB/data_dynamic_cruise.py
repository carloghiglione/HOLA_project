import numpy as np

time_phases = [75, 175]
lam = 0.75
matrix = np.array([[0.,         0.,     0.80,       lam*0.80,   0.],
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

conversion_rate_listofmatrix = []
mat = np.array([[0.1,   0.1,  0.2,  0.3],
                [0.5,   0.4,  0.3,  0.1],
                [0.1,   0.4,  0.3,  0.1],
                [0.5,   0.1,  0.1,  0.01],
                [0.01,  0.7,  0.1,  0.3]])
conversion_rate_listofmatrix.append([mat for _ in range(3)])
mat = np.array([[0.1,   0.1,  0.2,  0.3],
                [0.1,   0.2,  0.2,  0.3],
                [0.1,   0.4,  0.3,  0.1],
                [0.5,   0.1,  0.1,  0.01],
                [0.01,  0.4,  0.1,  0.3]])
conversion_rate_listofmatrix.append([mat for _ in range(3)])
mat = np.array([[0.5,   0.4,  0.3,  0.1],
                [0.2,   0.4,  0.3,  0.1],
                [0.1,   0.4,  0.3,  0.1],
                [0.2,   0.2,  0.1,  0.1],
                [0.4,   0.2,  0.1,  0.3]])
conversion_rate_listofmatrix.append([mat for _ in range(3)])

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
    "time_phases": time_phases,
    "meppp": meppp
}
