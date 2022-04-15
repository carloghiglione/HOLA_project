from P1_Base.••• import *   #al posto di ••• inserire le classi da importare
from P3_UCB_TS.•• import ••• #al posto di ••• importare il learner,
        # al posto di •• il file in cui si trova
from P1_Base.data_per_simulation_••• import data_dict # al posto di ••• mettere il pacchetto dati giusto

import numpy as np
import matplotlib.pyplot as plt
import sys

sys.stdout.write('\r' + str("Initializing simulation environment") + '\n')

env = Hyperparameters()
#IMPORTANTE:
#L'inizializzazione di env cambia da tipo di simulazione a tipo di simulazione, controlla le istruzioni nel
#relativo file; aka copia e incolla la prima riga del file data_per_simulation che hai importato

time_horizon = 200  #setta a piacere

day_profit = []
day_profit_per_prod = []
day_prices = np.zeros(5, dtype=int)
learner = Learner(env)  #metti qua il learner importato
best_prices = np.zeros(5, dtype=int)
for i in range(5):
    best_prices[i] = np.argmax(env.global_margin[i, :]*(env.global_conversion_rate[0][i,:]*env.pois_param[0] +
                                                        env.global_conversion_rate[1][i,:]*env.pois_param[1] + #nel caso di statico è pop_param
                                                        env.global_conversion_rate[2][i,:]*env.pois_param[2]))
best_prices = np.array(best_prices, dtype=int)
cl_profit = []

sys.stdout.write('\r' + str("Beginning simulation") + '\n')

for t in range(time_horizon):
    sys.stdout.write('\r' + "Simulation progress: " + f'{t*100/time_horizon} %')
    day = Day(env, day_prices)
    day.run_simulation()
    day_profit.append(day.profit)
    day_profit_per_prod.append(day.items_sold*day.website.margin)
    learner.update(day)
    day_prices = learner.pull_prices()

    cl_profit.append(day.run_clairvoyant_simulation(best_prices))
sys.stdout.flush()
sys.stdout.write('\r' + str("Simulation completed under time horizon t = ") + str(time_horizon) + str(" days") + '\n')

final_prices = day_prices
print(f'Final price configuration: {final_prices}')
plt.figure(0)
plt.plot(day_profit, color='red')
plt.plot(cl_profit, color='blue')
plt.legend(["TS", "Optimal"], loc='best')
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