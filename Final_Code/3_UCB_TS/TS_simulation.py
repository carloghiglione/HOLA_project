from Classes_fixed_params import *
from TS import Items_TS_Learner
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

time_horizon = 200

day_profit = []
day_profit_per_prod = []
day_prices = np.zeros(5, dtype=int)
learner = Items_TS_Learner(env)
best_prices = np.zeros(5, dtype=int)
for i in range(5):
    best_prices[i] = np.argmax(env.global_margin[i, :]*(env.global_conversion_rate[0][i,:]*env.pop_param[0] +
                                                        env.global_conversion_rate[1][i,:]*env.pop_param[1] +
                                                        env.global_conversion_rate[2][i,:]*env.pop_param[2]))
best_prices = np.array(best_prices, dtype=int)
cl_profit = []

for t in range(time_horizon):
    print(f'{t*100/time_horizon} %')
    day = Day(env, day_prices)
    day.run_simulation()
    day_profit.append(day.profit)
    day_profit_per_prod.append(day.items_sold*day.website.margin)
    learner.update(day)
    day_prices = learner.pull_prices()

    cl_profit.append(day.run_clairvoyant_simulation(best_prices))


final_prices = day_prices
print(f'Final price configuration: {final_prices}')
plt.figure(0)
plt.plot(day_profit, color='red')
plt.plot(cl_profit, color='blue')
plt.legend(["TS", "Optimal"], loc='best')
plt.title("Profit - simulation")
plt.xlabel("time [day]")
plt.ylabel("profit [euros]")
plt.tight_layout()
plt.show()

plt.figure(1)
plt.plot(np.cumsum(np.array(cl_profit) - np.array(day_profit)))
plt.title("Regret in single simulation")
plt.xlabel("time [day]")
plt.ylabel("regret [euros]")
plt.tight_layout()
plt.show()
