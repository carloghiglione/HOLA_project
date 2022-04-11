import numpy as np

from Classes import *

class greedy:
    def __init__(self, env: Hyperparameters, current_profit, initial_prices):
        self.current_profit = current_profit
        self.check_convergence = False
        self.prices = initial_prices
        self.env=env

    def step(self):
        greedy_profits= np.zeros(5)
        for i in range(5): #for every product we raise one price at a time
            new_prices = self.prices
            if new_prices[i] != 3:
                new_prices[i] += 1
                day=Day(self.env, new_prices)
                day.run_simulation()
                greedy_profits[i] = day.profit
            else:
                self.check_convergence = True
                print("Convergence of greedy reached")

        greedy_delta = np.diff(greedy_profits, np.ones(5) * self.current_profit)
        if(np.max(greedy_delta) > 0 & self.check_convergence==False):
            best_product = np.argmax(greedy_delta)
            self.prices[best_product] +=1
        else:
            self.check_convergence = True
            print("Convergence of greedy reached")



