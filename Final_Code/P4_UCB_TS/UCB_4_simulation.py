import copy
import sys

time_horizon = 50
seed = 17021890

sys.stdout.write('\r' + str("Initializing simulation environment"))
from P1_Base.Classes_base import *
from UCB_4 import Items_UCB_Learner
from P1_Base.Price_puller import pull_prices
import numpy as np
import matplotlib.pyplot as plt
from P1_Base.data_cruise import data_dict
env = Hyperparameters(data_dict["tr_prob"], data_dict["dir_par"], data_dict["pois_par"],
                      data_dict["conv_rate"], data_dict["margin"], data_dict["meppp"])

sys.stdout.write(str(": Done") + '\n')

np.random.seed(seed)

day_profit = []
day_normalized_profit = []
day_profit_per_prod = []
day_prices = np.zeros(5, dtype=int)
learner = Items_UCB_Learner(copy.deepcopy(env))
printer = str(('\r' + str("Finding Clairvoyant solution")))
best_prices = pull_prices(env=copy.deepcopy(env), conv_rates=copy.deepcopy(env.global_conversion_rate),
                          alpha=copy.deepcopy(env.dir_params), n_buy=copy.deepcopy(env.mepp),
                          trans_prob=copy.deepcopy(env.global_transition_prob), print_message=printer)
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
    day_prices = learner.pull_prices(copy.deepcopy(env), print_message)
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
