from Classes_fixed_params import *
from Greedy import greedy

#enviroment definition
from data_per_simulation_0 import data_dict
env = Hyperparameters(data_dict["tr_prob"], data_dict["dir_par"], data_dict["pois_par"],
                      data_dict["conv_rate"], data_dict["margin"])

#best prices clairvoyant
best_prices = np.zeros(5, dtype=int)
for i in range(5):
    best_prices[i] = np.argmax(env.global_margin[i, :]*(env.global_conversion_rate[0][i,:]*env.pop_param[0] +
                                                        env.global_conversion_rate[1][i,:]*env.pop_param[1] +
                                                        env.global_conversion_rate[2][i,:]*env.pop_param[2]))
best_prices = np.array(best_prices, dtype=int)

#testing greedy
starting_prices=np.zeros(5,dtype=int) #all prices start at minimum
starting_day = Day(env,starting_prices)
starting_day.run_simulation()

g=greedy(env, starting_day.profit , starting_prices)
while (g.check_convergence==False) :
          g.step()

greedy_best_prices=g.prices
print("Best prices indexes by greedy algorithm:",greedy_best_prices)
print("Best prices indexes by clairvoyant algorithm:",best_prices)
