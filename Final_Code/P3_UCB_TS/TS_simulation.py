import copy
import sys
sys.path.insert(0, '..')

n_user_cl = 1500
time_horizon = 50
n_users_MC = 250

sys.stdout.write('\r' + str("Initializing simulation environment"))
from P1_Base.Classes_base import *
from TS import Items_TS_Learner
from P1_Base.MC_simulator import pull_prices
import numpy as np
import matplotlib.pyplot as plt
from P1_Base.data_base import data_dict
env = Hyperparameters(data_dict["tr_prob"], data_dict["dir_par"], data_dict["pois_par"],
                      data_dict["conv_rate"], data_dict["margin"])

sys.stdout.write(str(": Done") + '\n')

day_profit = []
day_normalized_profit = []
day_profit_per_prod = []
day_prices = np.zeros(5, dtype=int)
learner = Items_TS_Learner(copy.deepcopy(env))
printer = str(('\r' + str("Finding Clairvoyant solution")))
best_prices = pull_prices(env=copy.deepcopy(env), conv_rates=copy.deepcopy(env.global_conversion_rate),
                          alpha=copy.deepcopy(env.dir_params), n_buy=copy.deepcopy(env.mepp),
                          trans_prob=copy.deepcopy(env.global_transition_prob), n_users_pt=n_user_cl,
                          print_message=printer)
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
    day_normalized_profit.append(day.profit/np.sum(day.n_users))
    # day_profit_per_prod.append(np.array(day.items_sold*day.website.margin, dtype=float))
    learner.update(day)
    day_prices = learner.pull_prices(copy.deepcopy(env), print_message, n_users_pt=n_users_MC)
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
plt.plot(cl, color='green')
plt.legend(["TS", "Optimal - mean"], loc='best')
plt.title("Profit - simulation")
plt.xlabel("time [day]")
plt.ylabel("profit [euros]")
plt.tight_layout()
plt.show()

plt.figure(1)
plt.plot(np.cumsum(np.array(cl_line) - np.array(day_profit)))
plt.title("Regret in single simulation")
plt.xlabel("time [day]")
plt.ylabel("regret [euros]")
plt.tight_layout()
plt.show()
