import copy
import sys

n_user_cl = 1500
time_horizon = 50
n_users_MC = 250

sys.stdout.write('\r' + str("Initializing simulation environment"))
from Classes_dynamic import *
from CD_UCB import Items_UCB_Learner
from P1_Base.MC_simulator import pull_prices
import numpy as np
import matplotlib.pyplot as plt
from data_dynamic import data_dict
env = Hyperparameters(data_dict["tr_prob"], data_dict["dir_par"], data_dict["pois_par"],
                      data_dict["conv_rate"], data_dict["margin"], data_dict["time_phases"])

sys.stdout.write(str(": Done") + '\n')

day_profit = []
day_normalized_profit = []
cl_profit = []
day_prices = np.zeros(5, dtype=int)
learner = Items_UCB_Learner(copy.deepcopy(env))


best_prices = [np.zeros(5, dtype=int) for i in range(len(env.phases)+1)]
for p in range(len(env.phases)+1):
    printer = str(('\r' + str("Finding Clairvoyant solution for phase ") + str(p+1)))
    best_prices[p] = pull_prices(env=copy.deepcopy(env), conv_rates=copy.deepcopy(env.global_conversion_rate[p]),
                                 alpha=copy.deepcopy(env.dir_params), n_buy=copy.deepcopy(env.mepp),
                                 trans_prob=copy.deepcopy(env.global_transition_prob), n_users_pt=n_user_cl,
                                 print_message=printer)
sys.stdout.write('\r' + str("Finding Clairvoyant solution: Done") + '\n')
print(f'Clairvoyant price configuration: {best_prices}')

sys.stdout.write('\r' + str("Beginning simulation") + '\n')

phase_big = [-1]
for i in range(len(env.phases)):
    phase_big.append(env.phases[i])
phase_big.append(np.inf)

for t in range(time_horizon):
    print_message = str('\r' + "Simulation progress: " + f'{t * 100 / time_horizon} %')
    day = Day(copy.deepcopy(env), day_prices)
    day.run_simulation()
    day_profit.append(day.profit)
    day_normalized_profit.append(day.profit / np.sum(day.n_users))
    # day_profit_per_prod.append(np.array(day.items_sold*day.website.margin, dtype=float))
    learner.update(day)
    day_prices = learner.pull_prices(copy.deepcopy(env), print_message, n_users_pt=n_users_MC)

    index = -1
    for i in range(len(env.phases) + 1):
        if phase_big[i] < env.t <= phase_big[i + 1]:
            index = i
    cl_profit.append(day.run_clairvoyant_simulation(best_prices[index]))
    env.t += 1

sys.stdout.flush()
sys.stdout.write('\r' + str("Simulation completed under time horizon t = ") + str(time_horizon) + str(" days") + '\n')

cl_line = np.zeros(time_horizon)
if len(env.phases) > 0:
    for i in range(len(env.phases)):
        if i == 0:
            iw_mean = np.mean(cl_profit[:env.phases[i]])
            cl_line[0:env.phases[i]] = iw_mean*np.ones(env.phases[i])
        else:
            iw_mean = np.mean(cl_profit[env.phases[i-1]:env.phases[i]])
            cl_line[env.phases[i-1]:env.phases[i]] = iw_mean * np.ones(env.phases[i] - env.phases[i-1])

        if i == (len(env.phases)-1):
            iw_mean = np.mean(cl_profit[env.phases[i]:])
            cl_line[env.phases[i]:] = iw_mean * np.ones(time_horizon - env.phases[i])
else:
    cl_line = np.mean(cl_profit)*np.ones(time_horizon)

plt.figure(0)
plt.plot(day_profit, color='red')
plt.plot(cl_line, color='blue')
plt.legend(["UCB", "Optimal"], loc='best')
plt.title("Profit - simulation")
plt.xlabel("time [day]")
plt.ylabel("profit [euros]")
plt.tight_layout()
plt.show()


print("Changes detected:")
for i in range(5):
    print(learner.learners[i].detections)
    print("***")
