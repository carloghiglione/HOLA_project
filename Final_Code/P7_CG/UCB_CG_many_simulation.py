import copy
import sys

time_horizon = 50
context_window = 14
n_trials = 10
seed = 17021890

sys.stdout.write('\r' + str("Initializing simulation environment"))
from Classes_CG import *
from UCB_CG import CG_Learner
from Price_puller_CG import pull_prices, expected_profits
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

profits_for_config = expected_profits(env=copy.deepcopy(env), conv_rates=copy.deepcopy(env.global_conversion_rate),
                                      alpha=copy.deepcopy(env.dir_params), n_buy=copy.deepcopy(env.mepp),
                                      trans_prob=copy.deepcopy(env.global_transition_prob),
                                      print_message=str(('\r' + str("Computing expected profits"))))

sys.stdout.write('\r' + str("Computing expected profits: Done") + '\n')
optimal_expected_profit = (profits_for_config[0][best_prices[0, 0, 0], best_prices[0, 0, 1], best_prices[0, 0, 2],
                                                 best_prices[0, 0, 3], best_prices[0, 0, 4]] *
                           float(np.sum(env.pois_param[0])) +
                           profits_for_config[1][
                               best_prices[1, 0, 0], best_prices[1, 0, 1], best_prices[1, 0, 2], best_prices[1, 0, 3],
                               best_prices[1, 0, 4]] * float(env.pois_param[1][0]) +
                           profits_for_config[2][
                               best_prices[1, 1, 0], best_prices[1, 1, 1], best_prices[1, 1, 2], best_prices[1, 1, 3],
                               best_prices[1, 1, 4]] * float(env.pois_param[1][1]))

profits = []
profits_cl = []
final_prices = []
expected_prof = []
regret = []
pseudoregret = []

for sim in range(n_trials):
    day_profit = []
    day_profit_per_prod = []
    day_prices = np.zeros(shape=(2, 2, 5), dtype=int)
    learner = CG_Learner(copy.deepcopy(env), context_window=context_window)
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
        e_prof[t] = (profits_for_config[0][
                               day_prices[0, 0, 0], day_prices[0, 0, 1], day_prices[0, 0, 2], day_prices[0, 0, 3],
                               day_prices[0, 0, 4]] * float(env.pois_param[0][0]) +
                     profits_for_config[0][
                         day_prices[0, 1, 0], day_prices[0, 1, 1], day_prices[0, 1, 2], day_prices[0, 1, 3],
                         day_prices[0, 1, 4]] * float(env.pois_param[0][1]) +
                     profits_for_config[1][
                               day_prices[1, 0, 0], day_prices[1, 0, 1], day_prices[1, 0, 2], day_prices[1, 0, 3],
                               day_prices[1, 0, 4]] * float(env.pois_param[1][0]) +
                     profits_for_config[2][
                         day_prices[1, 1, 0], day_prices[1, 1, 1], day_prices[1, 1, 2], day_prices[1, 1, 3],
                         day_prices[1, 1, 4]] * float(env.pois_param[1][1]))
        learner.update(day)
        day_prices = learner.pull_prices(print_message)
        cl_profit.append(day.run_clairvoyant_simulation(best_prices))

    sys.stdout.write('\r' + "Simulation n." + str(sim + 1) + ": 100%" + '\n')
    sys.stdout.flush()

    final_prices.append(copy.deepcopy(day_prices))

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
    sys.stdout.write('\r' + str("Simulation n." + str(sim+1) + " completed, "
                                + f'final price configuration: {final_prices[sim]}' + '\n'))
    print("context history:")
    print(np.array(learner.context_history))


profits = np.array(profits)
profits_cl = np.array(profits_cl)
expected_prof = np.array(expected_prof)
regret = np.array(regret)
pseudoregret = np.array(pseudoregret)

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
plt.fill_between(range(time_horizon), mean_psereg - sd_psereg, mean_psereg + sd_psereg, alpha=0.4, color='blue')
plt.title("Cumulative Pseudo-Regret")
plt.xlabel("time [day]")
plt.ylabel("regret [euros]")
plt.tight_layout()
plt.show()

total_regret = np.mean(np.sum(regret, axis=0))
print("=============================")
print("=============================")
print("Mean total regret = " + str(total_regret))
print("=============================")
print("Optimal profit = " + str(optimal_expected_profit))

