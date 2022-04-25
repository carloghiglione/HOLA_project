import sys
sys.path.insert(0, '..')

random_environment = False
time_horizon = 100

if random_environment:
    sys.stdout.write('\r' + str("Initializing random simulation environment"))
    from P1_Base.Classes_random_parameters import *
    from UCB import Items_UCB_Learner
    import numpy as np
    import matplotlib.pyplot as plt
    from P1_Base.random_data import data_dict
    env = Hyperparameters(data_dict["tr_prob"], data_dict["dir_hpar"], data_dict["dir_poi"],
                          data_dict["pois_poi"], data_dict["conv_rate"], data_dict["margin"])

else:
    sys.stdout.write('\r' + str("Initializing simulation environment"))
    from P1_Base.Classes_base import *
    from UCB import Items_UCB_Learner
    import numpy as np
    import matplotlib.pyplot as plt
    from P1_Base.data_base import data_dict
    env = Hyperparameters(data_dict["tr_prob"], data_dict["dir_par"], data_dict["pois_par"],
                          data_dict["conv_rate"], data_dict["margin"])

sys.stdout.write(str(": Done") + '\n')

day_profit = []
day_profit_per_prod = []
day_prices = np.zeros(5, dtype=int)
learner = Items_UCB_Learner(env)

best_prices = np.zeros(5, dtype=int)
for i in range(5):
    best_prices[i] = np.argmax(env.global_margin[i, :]*(env.global_conversion_rate[0][i, :]*env.pois_param[0] +
                                                        env.global_conversion_rate[1][i, :]*env.pois_param[1] +
                                                        env.global_conversion_rate[2][i, :]*env.pois_param[2]))
best_prices = np.array(best_prices, dtype=int)
cl_profit = []
sys.stdout.write('\r' + str("Beginning simulation") + '\n')


cr = []

for t in range(time_horizon):
    sys.stdout.write('\r' + "Simulation progress: " + f'{t * 100 / time_horizon} %')
    day = Day(env, day_prices)
    day.run_simulation()
    day_profit.append(day.profit)
    day_profit_per_prod.append(day.items_sold*day.website.margin)
    learner.update(day)
    day_prices = learner.pull_prices()

    cl_profit.append(day.run_clairvoyant_simulation(best_prices))

    cr.append(learner.learners[0].means[3])

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

plt.figure(1)
plt.plot(np.cumsum(np.array(cl_profit) - np.array(day_profit)))
plt.title("Regret in single simulation")
plt.xlabel("time [day]")
plt.ylabel("regret [euros]")
plt.tight_layout()
plt.show()


#plt.figure(2)
#plt.plot(np.diff(cr)[5:])
#plt.title("CR")
#plt.xlabel("time [day]")
#plt.ylabel("CR")
#plt.tight_layout()
#plt.show()
#
#print(cr)