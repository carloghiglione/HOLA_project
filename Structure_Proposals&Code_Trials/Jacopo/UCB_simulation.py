from Classes import *
from UCB_trial import Items_UCB_Learner
import numpy as np
import matplotlib.pyplot as plt


matrix = np.array([[0.,   0.5,   0.5,   0.,    0.],
                   [0.,   0.,    0.5,   0.5,   0.],
                   [0.,   0.,    0.,    0.5,   0.5],
                   [0.5,  0.,    0.,    0.,    0.5],
                   [0.5,  0.5,   0.,    0.,    0.]])
transition_prob_listofmatrix = [matrix for i in range(3)]
vec = 100*np.ones(6)
dir_params_listofvector = [vec for i in range(3)]
pois_param_vector = [500 for i in range(3)]
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

env = Hyperparameters(transition_prob_listofmatrix,dir_params_listofvector, pois_param_vector, conversion_rate_listofmatrix, margin_matrix)

time_horizon = 100

day_profit = []
day_profit_per_prod = []
day_prices = np.zeros(5, dtype=int)
learner = Items_UCB_Learner(env)

for t in range(time_horizon):
    print(f'{t*100/time_horizon} %')
    day = Day(env, day_prices)
    day.run_simulation()
    day_profit.append(day.profit)
    day_profit_per_prod.append(day.items_sold*day.website.margin)
    learner.update(day)
    day_prices = learner.pull_prices()

final_prices = day_prices
plt.plot(day_profit)
plt.show()