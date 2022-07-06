import copy
import sys

time_horizon = 90
n_trials = 10
seed = 17021890

sys.stdout.write('\r' + str("Initializing simulation environment"))
from Classes_dynamic import *
from CD_UCB import Abrupt_Items_UCB_Learner
from P1_Base.Price_puller import pull_prices, expected_profits
import numpy as np
import matplotlib.pyplot as plt
from data_dynamic import data_dict
from copy import deepcopy as cdc
env = Hyperparameters(data_dict["tr_prob"], data_dict["dir_par"], data_dict["pois_par"],
                      data_dict["conv_rate"], data_dict["margin"], data_dict["time_phases"], data_dict["meppp"])

sys.stdout.write(str(": Done") + '\n')

np.random.seed(seed)

best_prices = [np.zeros(5, dtype=int) for _ in range(len(env.phases)+1)]
e_profits = [np.zeros(shape=(4, 4, 4, 4, 4), dtype=float) for _ in range(len(env.phases)+1)]
optimal_expected_profit = np.zeros(len(env.phases)+1, dtype=float)
for p in range(len(env.phases)+1):
    printer = str(('\r' + str("Finding Clairvoyant solution for phase ") + str(p+1)))
    best_prices[p] = pull_prices(env=cdc(env), conv_rates=cdc(env.global_conversion_rate[p]),
                                 alpha=cdc(env.dir_params), n_buy=cdc(env.mepp),
                                 trans_prob=cdc(env.global_transition_prob), print_message=printer)
    e_profits[p] = expected_profits(env=copy.deepcopy(env), conv_rates=copy.deepcopy(env.global_conversion_rate[p]),
                                    alpha=copy.deepcopy(env.dir_params), n_buy=copy.deepcopy(env.mepp),
                                    trans_prob=copy.deepcopy(env.global_transition_prob),
                                    print_message=str(('\r' + str("Computing expected profits"))))*float(np.sum(env.pois_param))
    optimal_expected_profit[p] = e_profits[p][best_prices[p][0],
                                              best_prices[p][1],
                                              best_prices[p][2],
                                              best_prices[p][3],
                                              best_prices[p][4]]
sys.stdout.write('\r' + str("Finding Clairvoyant solution: Done") + '\n')
print(f'Clairvoyant price configuration for each phase: {best_prices}')
sys.stdout.write('\r' + str("Computing expected profits: Done") + '\n')

profits = []
profits_cl = []
final_prices = []
expected_prof = []
pseudoregret = []


phase_big = [-1]
for i in range(len(env.phases)):
    phase_big.append(env.phases[i])
phase_big.append(np.inf)

for sim in range(n_trials):
    day_profit = []
    day_profit_per_prod = []
    day_prices = np.zeros(5, dtype=int)
    learner = Abrupt_Items_UCB_Learner(env)
    cl_profit = []
    psereg = np.zeros(time_horizon, dtype=float)
    e_prof = np.zeros(time_horizon, dtype=float)

    print("=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*")
    sys.stdout.write('\r' + str("Simulation n.") + str(sim+1) + str(" out of ") + str(n_trials) + '\n')
    np.random.seed(sim*seed)

    for t in range(time_horizon):
        print_message = str('\r' + "Simulation n." + str(sim+1) + ": " + f'{t * 100 / time_horizon} %')
        day = Day(copy.deepcopy(env), day_prices)
        day.run_simulation()
        day_profit.append(day.profit)
        # day_profit_per_prod.append(np.array(day.items_sold*day.website.margin, dtype=float))
        learner.update(day)
        day_prices = learner.pull_prices(copy.deepcopy(env), print_message)

        index = -1
        for i in range(len(env.phases) + 1):
            if phase_big[i] < env.t <= phase_big[i + 1]:
                index = i
        cl_profit.append(day.run_clairvoyant_simulation(best_prices[index]))
        e_prof[t] = e_profits[index][day_prices[0], day_prices[1], day_prices[2], day_prices[3], day_prices[4]]
        psereg[t] = optimal_expected_profit[index] - e_prof[t]

        env.t += 1

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
    pseudoregret.append(np.cumsum(psereg))
    sys.stdout.write('\r' + str("Simulation n." + str(sim+1) + " completed, "
                                + f'final price configuration: {final_prices[sim]}' + '\n'))



profits = np.array(profits)
if len(profits.shape) == 3:
    profits = profits[:, :, 0]

profits_cl = np.array(profits_cl)
if len(profits_cl.shape) == 3:
    profits_cl = profits_cl[:, :, 0]

expected_prof = np.array(expected_prof)
if len(expected_prof.shape) == 3:
    expected_prof = expected_prof[:, :, 0]

pseudoregret = np.array(pseudoregret)
if len(pseudoregret.shape) == 3:
    pseudoregret = pseudoregret[:, :, 0]

# viene salvata come lista di vettori colonna, che una volta messo come matrice ha una dimensione extra,
# è più semplice ridurre qui così non sfasiamo la struttura di un trial per riga
#profits = profits[:, :, 0]
#profits_cl = profits_cl[:, :, 0]
#regret = regret[:, :, 0]

mean_prof = np.mean(profits, axis=0)
sd_prof = np.std(profits, axis=0)

cl_line = np.zeros(time_horizon)
cl_band = np.zeros(time_horizon)
if len(env.phases) > 0:
    for i in range(len(env.phases)):
        if i == 0:
            iw_mean = np.mean(profits_cl[:, 0:env.phases[i]])
            cl_line[0:env.phases[i]] = copy.deepcopy(iw_mean*np.ones(env.phases[i]))
            iw_sd = np.std(profits_cl[:, 0:env.phases[i]])
            cl_band[0:env.phases[i]] = copy.deepcopy(iw_sd*np.ones(env.phases[i]))
        else:
            iw_mean = np.mean(profits_cl[:, env.phases[i - 1]:env.phases[i]])
            cl_line[env.phases[i - 1]:env.phases[i]] = copy.deepcopy(iw_mean * np.ones(env.phases[i] - env.phases[i - 1]))
            iw_sd = np.std(profits_cl[:, env.phases[i - 1]:env.phases[i]])
            cl_band[env.phases[i - 1]:env.phases[i]] = copy.deepcopy(iw_sd*np.ones(env.phases[i] - env.phases[i - 1]))

        if i == (len(env.phases)-1):
            iw_mean = np.mean(profits_cl[:, env.phases[i]:])
            cl_line[env.phases[i]:] = copy.deepcopy(iw_mean * np.ones(time_horizon - env.phases[i]))
            iw_sd = np.std(profits_cl[:, env.phases[i]:])
            cl_band[env.phases[i]:] = copy.deepcopy(iw_sd * np.ones(time_horizon - env.phases[i]))
else:
    cl_line = np.mean(profits_cl) * np.ones(time_horizon)
    cl_band = np.sd(profits_cl) * np.ones(time_horizon)

mean_psereg = np.mean(pseudoregret, axis=0)
sd_psereg = np.std(pseudoregret, axis=0)

plt.figure(0)
plt.plot(mean_prof, color='red')
plt.plot(cl_line, color='blue')
plt.fill_between(range(time_horizon), mean_prof - sd_prof, mean_prof + sd_prof, alpha=0.4, color='red')
plt.fill_between(range(time_horizon), cl_line - cl_band, cl_line + cl_band, alpha=0.4, color='blue')
plt.legend(["CD UCB", "Optimal"], loc='best')
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

total_pseudoregret = np.mean(np.sum(pseudoregret, axis=0))
print("=============================")
print("=============================")
print("Mean total regret = " + str(total_pseudoregret))
