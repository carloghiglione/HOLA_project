import copy
import sys

n_user_cl = 1500
seed = 17021890

sys.stdout.write('\r' + str("Initializing simulation environment"))
from P1_Base.Classes_base import *
from P1_Base.MC_simulator import pull_prices
from Greedy import Greedy
import numpy as np
import matplotlib.pyplot as plt
from P1_Base.data_base import data_dict
env = Hyperparameters(data_dict["tr_prob"], data_dict["dir_par"], data_dict["pois_par"],
                      data_dict["conv_rate"], data_dict["margin"])

sys.stdout.write(str(": Done") + '\n')

np.random.seed(seed)

# best prices clairvoyant
printer = str(('\r' + str("Finding Clairvoyant solution")))
best_prices = pull_prices(env=copy.deepcopy(env), conv_rates=copy.deepcopy(env.global_conversion_rate),
                          alpha=copy.deepcopy(env.dir_params), n_buy=copy.deepcopy(env.mepp),
                          trans_prob=copy.deepcopy(env.global_transition_prob), n_users_pt=n_user_cl,
                          print_message=printer)
sys.stdout.write('\r' + str("Finding Clairvoyant solution: Done") + '\n')

# testing greedy
starting_prices = np.zeros(5,dtype=int)  # all prices start at minimum
starting_day = Day(env, starting_prices)
starting_day.run_simulation()

g = Greedy(env, starting_day.profit/np.sum(starting_day.n_users), starting_prices, best_prices)
count = 1
while g.check_convergence == False:
    sys.stdout.write('\r' + "Running simulation: Step n." + str(count))
    g.step()
    count = count+1

greedy_best_prices = g.prices
print("Best prices indexes by greedy algorithm:", greedy_best_prices)
print("Best prices indexes by clairvoyant algorithm:", best_prices)


day_profit = g.day_profits
cl_profit = g.clairvoyant_profits
cl_mean = np.mean(cl_profit)
cl_line = cl_mean*np.ones(len(cl_profit))

plt.figure(0)
plt.plot(day_profit, color='red')
plt.plot(cl_line, color='blue')
plt.legend(["Greedy", "Optimal-mean"], loc='best')
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
