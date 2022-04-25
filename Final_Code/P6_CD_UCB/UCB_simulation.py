import sys
random_environment = True
time_horizon = 300


sys.stdout.write('\r' + str("Initializing simulation environment"))
from Classes_dynamic import *
from UCB import Items_UCB_Learner
import numpy as np
import matplotlib.pyplot as plt
from data_dynamic import data_dict

env = Hyperparameters(data_dict["tr_prob"], data_dict["dir_par"], data_dict["pois_par"],
                          data_dict["conv_rate"], data_dict["margin"], data_dict["time_phases"])



day_profit = []
day_profit_per_prod = []
day_prices = np.zeros(5, dtype=int)
learner = Items_UCB_Learner(env)

best_prices = [np.zeros(5, dtype=int) for i in range(len(env.phases)+1)]

for p in range(len(env.phases)+1):
    for i in range(5):
        best_prices[p][i] = np.argmax(env.global_margin[i, :]*(env.global_conversion_rate[p][0][i, :]*env.pois_param[0] +
                                                               env.global_conversion_rate[p][1][i, :]*env.pois_param[1] +
                                                               env.global_conversion_rate[p][2][i, :]*env.pois_param[2]))
cl_profit = []
sys.stdout.write('\r' + str("Beginning simulation") + '\n')

phase_big = [-1]
for i in range(len(env.phases)):
    phase_big.append(env.phases[i])
phase_big.append(np.inf)

for t in range(time_horizon):
    sys.stdout.write('\r' + "Simulation progress: " + f'{t * 100 / time_horizon} %')
    day = Day(env, day_prices)
    day.run_simulation()
    day_profit.append(day.profit)
    day_profit_per_prod.append(day.items_sold*day.website.margin)
    learner.update(day)
    day_prices = learner.pull_prices()

    index = -1

    for i in range(len(env.phases) + 1):
        if phase_big[i] < env.t <= phase_big[i + 1]:
            index = i

    cl_profit.append(day.run_clairvoyant_simulation(best_prices[index]))
    env.t += 1

sys.stdout.flush()
sys.stdout.write('\r' + str("Simulation completed under time horizon t = ") + str(time_horizon) + str(" days") + '\n')

final_prices = day_prices
print(f'Final price configuration: {final_prices}')
plt.figure(0)
plt.plot(day_profit, color='red')
plt.plot(cl_profit, color='blue')
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
