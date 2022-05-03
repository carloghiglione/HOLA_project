import copy
import sys

n_user_cl = 1500
time_horizon = 50
n_users_MC = 250
n_trials = 10
seed = 17021890

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

np.random.seed(seed)

printer = str(('\r' + str("Finding Clairvoyant solution")))
best_prices = pull_prices(env=copy.deepcopy(env), conv_rates=copy.deepcopy(env.global_conversion_rate),
                          alpha=copy.deepcopy(env.dir_params), n_buy=copy.deepcopy(env.mepp),
                          trans_prob=copy.deepcopy(env.global_transition_prob), n_users_pt=n_user_cl,
                          print_message=printer)
sys.stdout.write('\r' + str("Finding Clairvoyant solution: Done") + '\n')
print(f'Clairvoyant price configuration: {best_prices}')

profits = []
profits_cl = []
final_prices = []

for sim in range(n_trials):
    day_profit = []
    day_profit_per_prod = []
    day_prices = np.zeros(5, dtype=int)
    learner = Items_TS_Learner(env)
    cl_profit = []

    print("=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*")
    sys.stdout.write('\r' + str("Beginning simulation n.") + str(sim+1) + '\n')

    for t in range(time_horizon):
        print_message = str('\r' + "Simulation n." + str(sim+1) + ": " + f'{t * 100 / time_horizon} %')
        day = Day(copy.deepcopy(env), day_prices)
        day.run_simulation()
        day_profit.append(day.profit)
        learner.update(day)
        day_prices = learner.pull_prices(copy.deepcopy(env), print_message, n_users_pt=n_users_MC)
        cl_profit.append(day.run_clairvoyant_simulation(best_prices))

    sys.stdout.write('\r' + "Simulation n." + str(sim + 1) + ": 100%" + '\n')
    sys.stdout.flush()

    final_prices.append(day_prices)

    day_profit = np.array(day_profit)
    cl_profit = np.array(cl_profit)
    cl_profit.reshape(time_horizon)

    profits.append(copy.deepcopy(day_profit))
    profits_cl.append(copy.deepcopy(cl_profit))
    sys.stdout.write('\r' + str("Simulation n." + str(sim+1) + " completed, "
                                + f'final price configuration: {final_prices[sim]}' + '\n'))


profits = np.array(profits)
# viene salvata come lista di vettori colonna, che una volta messo come matrice ha una dimensione extra,
# è più semplice ridurre qui così non sfasiamo la struttura di un trial per riga
profits = profits[:, :, 0]
mean_prof = np.mean(profits, axis=0)
mean_prof_cl = np.mean(profits_cl)*np.ones(time_horizon)

regret = []
for i in range(n_trials):
    regret.append(np.cumsum(np.array(mean_prof_cl) - np.array(profits[i])))
mean_reg = np.mean(regret, axis=0)
sd_reg = np.std(regret, axis=0)

plt.figure(0)
plt.plot(mean_prof, color='red')
plt.plot(mean_prof_cl, color='blue')
plt.legend(["TS", "Optimal"], loc='best')
plt.title("Mean Profit - Simulations")
plt.xlabel("time [day]")
plt.ylabel("profit [euros]")
plt.tight_layout()
plt.show()

plt.figure(1)
plt.plot(mean_reg)
plt.fill_between(range(time_horizon), mean_reg - sd_reg, mean_reg + sd_reg, alpha=0.4)
plt.title("Mean cumulative regret")
plt.xlabel("time [day]")
plt.ylabel("regret [euros]")
plt.tight_layout()
plt.show()
