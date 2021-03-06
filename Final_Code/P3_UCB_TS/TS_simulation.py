import sys

time_horizon = 50
seed = 17021890

sys.stdout.write('\r' + str("Initializing simulation environment"))
from P1_Base.Classes_base import *
from TS import Items_TS_Learner
from P1_Base.Price_puller import pull_prices, expected_profits
import numpy as np
import matplotlib.pyplot as plt
from P1_Base.data_cruise import data_dict
env = Hyperparameters(data_dict["tr_prob"], data_dict["dir_par"], data_dict["pois_par"],
                      data_dict["conv_rate"], data_dict["margin"], mean_extra_purchases_per_product=data_dict["meppp"])

sys.stdout.write(str(": Done") + '\n')

np.random.seed(seed)

day_profit = []
day_normalized_profit = []
day_profit_per_prod = []
day_prices = 2*np.ones(5, dtype=int)
learner = Items_TS_Learner(copy.deepcopy(env))
printer = str(('\r' + str("Finding Clairvoyant solution")))
best_prices = pull_prices(env=copy.deepcopy(env), conv_rates=copy.deepcopy(env.global_conversion_rate),
                          alpha=copy.deepcopy(env.dir_params), n_buy=copy.deepcopy(env.mepp),
                          trans_prob=copy.deepcopy(env.global_transition_prob), print_message=printer)
sys.stdout.write('\r' + str("Finding Clairvoyant solution: Done") + '\n')
print(f'Clairvoyant price configuration: {best_prices}')

profits_for_config = expected_profits(env=copy.deepcopy(env), conv_rates=copy.deepcopy(env.global_conversion_rate),
                                      alpha=copy.deepcopy(env.dir_params), n_buy=copy.deepcopy(env.mepp),
                                      trans_prob=copy.deepcopy(env.global_transition_prob),
                                      print_message=str(('\r' + str("Computing expected profits"))))
sys.stdout.write('\r' + str("Computing expected profits: Done") + '\n')
profits_for_config = float(np.sum(env.pois_param))*profits_for_config
optimal_profit = profits_for_config[best_prices[0],
                                    best_prices[1],
                                    best_prices[2],
                                    best_prices[3],
                                    best_prices[4]]

print("==========")

sys.stdout.write('\r' + str("Beginning simulation") + '\n')

cl = []
e_prof = np.zeros(time_horizon, dtype=float)

for t in range(time_horizon):
    print_message = str('\r' + "Simulation progress: " + f'{t * 100 / time_horizon} %')
    day = Day(copy.deepcopy(env), day_prices)
    day.run_simulation()
    day_profit.append(day.profit)
    day_normalized_profit.append(day.profit/np.sum(day.n_users))
    # day_profit_per_prod.append(np.array(day.items_sold*day.website.margin, dtype=float))
    e_prof[t] = profits_for_config[day_prices[0], day_prices[1], day_prices[2], day_prices[3], day_prices[4]]
    learner.update(day)
    day_prices = learner.pull_prices(env, print_message)
    cl.append(day.run_clairvoyant_simulation(best_prices))

sys.stdout.flush()
sys.stdout.write('\r' + str("Simulation completed under time horizon t = ") + str(time_horizon) + str(" days") + '\n')

final_prices = day_prices
cl_mean = np.mean(cl)
cl_line = cl_mean*np.ones(time_horizon)
profit_with_optimal = np.array(cl, dtype=float)

print(f'Final price configuration: {final_prices}')
plt.figure(0)
plt.plot(day_profit, color='red')
plt.plot(cl_line, color='blue')
plt.plot(profit_with_optimal, color='green')
plt.legend(["TS", "Optimal - mean", "Rewards of Clairvoyant"], loc='best')
plt.title("Profit - simulation")
plt.xlabel("time [day]")
plt.ylabel("profit [euros]")
plt.tight_layout()
plt.show()

plt.figure(1)
plt.plot(np.cumsum(profit_with_optimal - day_profit), color='red')
plt.plot(np.cumsum(optimal_profit - e_prof), color='blue')
plt.legend(["Regret", "Pseudo Regret"], loc='best')
plt.title("Regret in single simulation")
plt.xlabel("time [day]")
plt.ylabel("regret [euros]")
plt.tight_layout()
plt.show()
