import numpy as np
import sys
sys.path.insert(0, '..')

random_environment = False
time_horizon = 100
n_trials = 50

seed = 27011999
np.random.seed(seed)

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


best_prices = np.zeros(5, dtype=int)
for i in range(5):
    best_prices[i] = np.argmax(env.global_margin[i, :] * (env.global_conversion_rate[0][i, :] * env.pois_param[0] +
                                                          env.global_conversion_rate[1][i, :] * env.pois_param[1] +
                                                          env.global_conversion_rate[2][i, :] * env.pois_param[2]))
best_prices = np.array(best_prices, dtype=int)

profits = []
profits_cl = []
final_prices = []

for sim in range(n_trials):
    day_profit = []
    day_profit_per_prod = []
    day_prices = np.zeros(5, dtype=int)
    learner = Items_UCB_Learner(env)
    cl_profit = []

    print("=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*")
    sys.stdout.write('\r' + str("Beginning simulation n.") + str(sim+1) + '\n')

    for t in range(time_horizon):
        sys.stdout.write('\r' + "Simulation n." + str(sim+1) + ": " + f'{t * 100 / time_horizon} %')
        day = Day(env, day_prices)
        day.run_simulation()
        day_profit.append(day.profit)
        day_profit_per_prod.append(day.items_sold*day.website.margin)
        learner.update(day)
        day_prices = learner.pull_prices()

        cl_profit.append(day.run_clairvoyant_simulation(best_prices))

    sys.stdout.write('\r' + "Simulation n." + str(sim + 1) + ": 100%" + '\n')
    sys.stdout.flush()

    final_prices.append(day_prices)
    profits.append(day_profit)
    profits_cl.append(cl_profit)
    sys.stdout.write(
        '\r' + str(
            "Simulation n." + str(sim + 1) + " completed, " + f'final price configuration: {final_prices[sim]}' + '\n'))


mean_prof = np.mean(profits, axis=0)
mean_prof_cl = np.mean(profits_cl, axis=0)

regret = []
for i in range(n_trials):
    regret.append(np.cumsum(np.array(profits_cl[i]) - np.array(profits[i])))
mean_reg = np.mean(regret, axis=0)
sd_reg = np.std(regret, axis=0)

plt.figure(0)
plt.plot(mean_prof, color='red')
plt.plot(mean_prof_cl, color='blue')
plt.legend(["UCB", "Optimal"], loc='best')
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
