import numpy as np
from Classes import *

class greedy:
    def __init__(self, env: Hyperparameters, current_profit, initial_prices):
        self.profit = current_profit
        self.check_convergence = False
        self.best_prices = initial_prices


    def step(self):
        greedy_profits= [0,0,0,0,0] #profitti che abbiamo aumentando giorno per giorno il prezzo
        prices = self.best_prices
        for i in range(5):
            new_prices = prices
            if new_prices[i] != 3:
                new_prices[i] += 1
                day=Day(env, new_prices)
                day.run_simulation()
                greedy_profits[i] = day.profit


        if(np.max(greedy_profits) > current_profit):
            best_product = np.argmax(greedy_profits)
            self.best_prices[best_product] +=1
        else:
            self.check_convergence = True


#main
hyper = Hyperparameters(...)
starting_prices=[0,0,0,0,0]
starting_day = day(hyper,starting_prices)
starting_day.run_simulation()

g=greedy(hyper, starting_day.profit , starting_prices)

while (!g.check_convergence) :
          g.step()
