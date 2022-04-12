import numpy as np
import copy
from Classes_fixed_params import *

class greedy:
    def __init__(self, env: Hyperparameters, init_profit, initial_prices):
        self.current_profit = init_profit
        self.check_convergence = False
        self.prices = initial_prices
        self.env=env

    def step(self):
        greedy_profits= np.zeros(5,dtype=float)
        for i in range(5): #for every product we raise one price at a time
            new_prices = copy.deepcopy(self.prices)
            if new_prices[i] != 3:
                new_prices[i] += 1
                day=Day(self.env, new_prices)
                day.run_simulation()
                greedy_profits[i] = day.profit
            else:
                greedy_profits[i]=self.current_profit

        greedy_delta=np.zeros(5,dtype=float)
        for i in range(5):
            greedy_delta[i]=greedy_profits[i]-self.current_profit

        if(np.max(greedy_delta) > 0):
            best_product = np.argmax(greedy_delta)
            self.prices[best_product] +=1
            self.current_profit=greedy_profits[best_product]
        else:
            self.check_convergence = True
            print("Convergence of greedy reached")

# QUESTION: should we use number of users -> normalized profits?

