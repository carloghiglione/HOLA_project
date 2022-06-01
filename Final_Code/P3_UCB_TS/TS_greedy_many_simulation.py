import copy
import sys

time_horizon = 30
n_trials = 5
seed = 17021890

sys.stdout.write('\r' + str("Initializing simulation environment"))
from P1_Base.Classes_base import *
from TS_greedy import Items_TS_Learner
from P1_Base.Price_puller import greedy_pull_price_exact, expected_profits, pull_prices
import numpy as np
import matplotlib.pyplot as plt
from P1_Base.data_cruise import data_dict
env = Hyperparameters(data_dict["tr_prob"], data_dict["dir_par"], data_dict["pois_par"],
                      data_dict["conv_rate"], data_dict["margin"], mean_extra_purchases_per_product=data_dict["meppp"])

sys.stdout.write(str(": Done") + '\n')

np.random.seed(seed)

printer = str(('\r' + str("Finding Clairvoyant solution")))
best_prices = greedy_pull_price_exact(env=copy.deepcopy(env), conv_rates=copy.deepcopy(env.global_conversion_rate),
                                      alpha=copy.deepcopy(env.dir_params), n_buy=copy.deepcopy(env.mepp),
                                      trans_prob=copy.deepcopy(env.global_transition_prob))
actual_best_prices = pull_prices(env=copy.deepcopy(env), conv_rates=copy.deepcopy(env.global_conversion_rate),
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
optimal_expected_profit = profits_for_config[best_prices[0],
                                             best_prices[1],
                                             best_prices[2],
                                             best_prices[3],
                                             best_prices[4]]
full_optimal_expected_profit = profits_for_config[actual_best_prices[0],
                                                  actual_best_prices[1],
                                                  actual_best_prices[2],
                                                  actual_best_prices[3],
                                                  actual_best_prices[4]]

profits = []
profits_cl = []
final_prices = []
expected_prof = []
regret = []
pseudoregret = []
true_pseudoregret = []

for sim in range(n_trials):
    day_profit = []
    day_profit_per_prod = []
    day_prices = np.zeros(5, dtype=int)
    learner = Items_TS_Learner(env)
    cl_profit = []
    e_prof = np.zeros(time_horizon, dtype=float)

    print("=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*")
    sys.stdout.write('\r' + str("Simulation n.") + str(sim+1) + str(" out of ") + str(n_trials) + '\n')
    np.random.seed(sim*seed)

    for t in range(time_horizon):
        print_message = str('\r' + "Simulation n." + str(sim+1) + ": " + f'{t * 100 / time_horizon} %')
        day = Day(copy.deepcopy(env), day_prices)
        day.run_simulation()
        day_profit.append(day.profit)
        e_prof[t] = profits_for_config[day_prices[0], day_prices[1], day_prices[2], day_prices[3], day_prices[4]]
        learner.update(day)
        day_prices = learner.pull_prices(env)
        cl_profit.append(day.run_clairvoyant_simulation(best_prices))
        sys.stdout.write(print_message)

    sys.stdout.write('\r' + "Simulation n." + str(sim + 1) + ": 100%" + '\n')
    sys.stdout.flush()

    final_prices.append(day_prices)

    day_profit = np.array(day_profit)
    day_profit.reshape(time_horizon)
    cl_profit = np.array(cl_profit)
    cl_profit.reshape(time_horizon)
    e_prof = np.array(e_prof)
    e_prof.reshape(time_horizon)

    profits.append(copy.deepcopy(day_profit))
    profits_cl.append(copy.deepcopy(cl_profit))
    expected_prof.append(copy.deepcopy(e_prof))
    regret.append(np.cumsum(cl_profit - day_profit))
    pseudoregret.append(np.cumsum(optimal_expected_profit - e_prof))
    true_pseudoregret.append(np.cumsum(full_optimal_expected_profit - e_prof))
    sys.stdout.write('\r' + str("Simulation n." + str(sim+1) + " completed, "
                                + f'final price configuration: {final_prices[sim]}' + '\n'))


profits = np.array(profits)
profits_cl = np.array(profits_cl)
expected_prof = np.array(expected_prof)
regret = np.array(regret)
pseudoregret = np.array(pseudoregret)
true_pseudoregret = np.array(true_pseudoregret)

# viene salvata come lista di vettori colonna, che una volta messo come matrice ha una dimensione extra,
# è più semplice ridurre qui così non sfasiamo la struttura di un trial per riga
#profits = profits[:, :, 0]
#profits_cl = profits_cl[:, :, 0]
#regret = regret[:, :, 0]

mean_prof = np.mean(profits, axis=0)
sd_prof = np.std(profits, axis=0)
mean_prof_cl = np.mean(profits_cl)*np.ones(time_horizon)
sd_prof_cl = np.std(profits_cl)*np.ones(time_horizon)

mean_reg = np.mean(regret, axis=0)
sd_reg = np.std(regret, axis=0)

mean_psereg = np.mean(pseudoregret, axis=0)
sd_psereg = np.std(pseudoregret, axis=0)
true_mean_psereg = np.mean(true_pseudoregret, axis=0)
true_sd_psereg = np.std(true_pseudoregret, axis=0)

plt.figure(0)
plt.plot(mean_prof, color='red')
plt.plot(mean_prof_cl, color='blue')
plt.fill_between(range(time_horizon), mean_prof - sd_prof, mean_prof + sd_prof, alpha=0.4, color='red')
plt.fill_between(range(time_horizon), mean_prof_cl - sd_prof_cl, mean_prof_cl + sd_prof_cl, alpha=0.4, color='blue')
plt.legend(["TS", "Optimal"], loc='best')
plt.title("Mean Profits")
plt.xlabel("time [day]")
plt.ylabel("profit [euros]")
plt.tight_layout()
plt.show()

plt.figure(1)
plt.plot(mean_psereg, color='blue')
plt.plot(true_mean_psereg, color='red')
plt.fill_between(range(time_horizon), mean_psereg - sd_psereg, mean_psereg + sd_psereg, alpha=0.4, color='blue')
plt.fill_between(range(time_horizon), true_mean_psereg - true_sd_psereg, true_mean_psereg + true_sd_psereg, alpha=0.4, color='red')
plt.legend(["PR greedy", "PR true"], loc='best')
plt.title("Cumulative Pseudo-Regret")
plt.xlabel("time [day]")
plt.ylabel("regret [euros]")
plt.tight_layout()
plt.show()

total_regret = np.mean(np.sum(regret, axis=0))
print("=============================")
print("=============================")
print("Mean total regret = " + str(total_regret))
