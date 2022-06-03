import copy
import sys

context_window = 15
time_horizon = 30
seed = 17021890

sys.stdout.write('\r' + str("Initializing simulation environment"))
from Classes_CG import *
from UCB_CG import CG_Learner
from Price_puller_CG import pull_prices
import numpy as np
import matplotlib.pyplot as plt
from data_CG import data_dict
env = Hyperparameters(transition_prob_listofmatrix=data_dict["tr_prob"],
                      dir_params_listofvector=data_dict["dir_par"],
                      pois_param_vector=data_dict["pois_par"],
                      conversion_rate_listofmatrix=data_dict["conv_rate"],
                      margin_matrix=data_dict["margin"],
                      feat_ass=data_dict["feat_ass"],
                      mean_extra_purchases_per_product=data_dict["meppp"])

sys.stdout.write(str(": Done") + '\n')

np.random.seed(seed)

day_profit = []
day_normalized_profit = []
day_profit_per_prod = []
day_prices = np.zeros(shape=(2, 2, 5), dtype=int)
learner = CG_Learner(copy.deepcopy(env), context_window=context_window)
best_prices = -1*np.ones(shape=(2, 2, 5), dtype=int)
best_prices_temp = [-1*np.ones(shape=5, dtype=float) for _ in range(3)]
for ty in range(3):
    printer = str(('\r' + str("Finding Clairvoyant solution for type n." + str(ty +1))))
    best_prices_temp[ty] = pull_prices(env=copy.deepcopy(env), conv_rates=copy.deepcopy(env.global_conversion_rate[ty]),
                              alpha=copy.deepcopy(env.dir_params[ty]), n_buy=copy.deepcopy(env.mepp[ty]),
                              trans_prob=copy.deepcopy(env.global_transition_prob), print_message=printer)
for f1 in range(2):
    for f2 in range(2):
        best_prices[f1, f2, :] = best_prices_temp[env.feature_associator[f1][f2]]
sys.stdout.write('\r' + str("Finding Clairvoyant solution: Done") + '\n')
print(f'Clairvoyant price configuration: {best_prices}')

print("==========")

sys.stdout.write('\r' + str("Beginning simulation") + '\n')


cl = []

for t in range(time_horizon):
    print_message = str('\r' + "Simulation progress: " + f'{t * 100 / time_horizon} %')
    day = Day(copy.deepcopy(env), day_prices)
    day.run_simulation()
    day_profit.append(day.profit)
    day_normalized_profit.append(day.profit / np.sum(day.n_users))
    # day_profit_per_prod.append(np.array(day.items_sold*day.website.margin, dtype=float))
    learner.update(day)
    day_prices = learner.pull_prices(print_message)
    cl.append(day.run_clairvoyant_simulation(best_prices))


sys.stdout.flush()
sys.stdout.write('\r' + str("Simulation completed under time horizon t = ") + str(time_horizon) + str(" days") + '\n')

final_prices = day_prices
cl_mean = np.mean(cl)
cl_line = cl_mean*np.ones(time_horizon)

print(f'Final price configuration: {final_prices}')
plt.figure(0)
plt.plot(day_profit, color='red')
plt.plot(cl_line, color='blue')
plt.legend(["UCB", "Optimal - mean"], loc='best')
plt.title("Profit - simulation")
plt.xlabel("time [day]")
plt.ylabel("profit [euros]")
plt.tight_layout()
plt.show()

plt.figure(1)
plt.plot(np.cumsum(cl_mean - day_profit))
plt.title("Regret in single simulation")
plt.xlabel("time [day]")
plt.ylabel("regret [euros]")
plt.tight_layout()
plt.show()

for contexts in range(len(learner.context_history)):
    print(learner.context_history[contexts])
