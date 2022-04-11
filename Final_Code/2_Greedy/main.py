from Classes import *
from Greedy_algo import greedy

#enviroment definition
transition_prob_listofmatrix=...
dir_params_listofvector=...
pois_param_vector=...
conversion_rate_listofmatrix=...
margin_matr=...
hyper = Hyperparameters(transition_prob_listofmatrix, dir_params_listofvector, pois_param_vector, conversion_rate_listofmatrix, margin_matr)


#testing greedy
starting_prices=[0,0,0,0,0]
starting_day = Day(hyper,starting_prices)
starting_day.run_simulation()

g=greedy(hyper, starting_day.profit , starting_prices)
while (g.check_convergence==False) :
          g.step()

greedy_best_prices=g.prices
print("Best prices indexes by greedy algorithm:",greedy_best_prices)
