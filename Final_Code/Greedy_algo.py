from Classes import *

class greedy:
    def __init__(self, env: Hyperparameters, current_profit, initial_prices):
        self.current_profit = current_profit
        self.check_convergence = False
        self.best_prices = initial_prices
        self.env=env

    def step(self):
        greedy_profits= [0,0,0,0,0]
        prices = self.best_prices
        for i in range(5):
            new_prices = prices
            if new_prices[i] != 3:
                new_prices[i] += 1
                day=Day(self.env, new_prices)
                day.run_simulation()
                greedy_profits[i] = day.profit

        if(np.max(greedy_profits) > self.current_profit):
            best_product = np.argmax(greedy_profits)
            self.best_prices[best_product] +=1
        else:
            self.check_convergence = True
            print("Convergence of greedy reached")



