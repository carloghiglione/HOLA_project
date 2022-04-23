import sys
sys.path.insert(0, '..')

random_environment = False

from Greedy import greedy
if random_environment:
    sys.stdout.write('\r' + str("Initializing random simulation environment"))
    from P1_Base.Classes_random_parameters import *
    import numpy as np
    import matplotlib.pyplot as plt
    from P1_Base.random_data import data_dict
    env = Hyperparameters(data_dict["tr_prob"], data_dict["dir_hpar"], data_dict["dir_poi"],
                          data_dict["pois_poi"], data_dict["conv_rate"], data_dict["margin"])

else:
    sys.stdout.write('\r' + str("Initializing simulation environment"))
    from P1_Base.Classes_base import *
    import numpy as np
    import matplotlib.pyplot as plt
    from P1_Base.data_base import data_dict
    env = Hyperparameters(data_dict["tr_prob"], data_dict["dir_par"], data_dict["pois_par"],
                          data_dict["conv_rate"], data_dict["margin"])

sys.stdout.write(str(": Done") + '\n')


#best prices clairvoyant
best_prices = np.zeros(5, dtype=int)
for i in range(5):
    best_prices[i] = np.argmax(env.global_margin[i, :]*(env.global_conversion_rate[0][i, :]*env.pois_param[0] +
                                                        env.global_conversion_rate[1][i, :]*env.pois_param[1] +
                                                        env.global_conversion_rate[2][i, :]*env.pois_param[2]))
best_prices = np.array(best_prices, dtype=int)

#testing greedy
starting_prices = np.zeros(5,dtype=int) #all prices start at minimum
starting_day = Day(env, starting_prices)
starting_day.run_simulation()

g = greedy(env, starting_day.profit, starting_prices, best_prices)
count = 1
while g.check_convergence==False:
    sys.stdout.write('\r' + "Running simulation: Step n." + str(count))
    g.step()
    count = count+1

greedy_best_prices = g.prices
print("Best prices indexes by greedy algorithm:", greedy_best_prices)
print("Best prices indexes by clairvoyant algorithm:", best_prices)


day_profit = g.day_profits
cl_profit = g.clairvoyant_profits

plt.figure(0)
plt.plot(day_profit, color='red')
plt.plot(cl_profit, color='blue')
plt.legend(["Greedy", "Optimal"], loc='best')
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

