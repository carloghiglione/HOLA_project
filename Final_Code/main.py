from Classes import *
from Greedy_algo import greedy

#enviroment definition
hyper = Hyperparameters(...)


#testing greedy
starting_prices=[0,0,0,0,0]
starting_day = Day(hyper,starting_prices)
starting_day.run_simulation()

g=greedy(hyper, starting_day.profit , starting_prices)

while (g.check_convergence==False) :
          g.step()
print(g.best_prices)