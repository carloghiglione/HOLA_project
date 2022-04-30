import numpy as np
import sys
sys.path.insert(0, '..')
from P1_Base.MC_simulator import pull_prices
from P1_Base.Classes_base import Hyperparameters

class Learner:

    def __init__(self, n_arms):
        self.t = 0
        self.n_arms = n_arms
        self.tot_clicks = np.zeros(n_arms)
        self.tot_sales = np.zeros(n_arms)

    # function to reset to initial condition
    def reset(self):
        self.__init__(self.n_arms)

    # perform one step of algo to select the arm that I pull, specific of the algorithm I use
    def act(self):
        pass

    # collect output of algo and append the reward
    def update(self, arm_pulled, sales, clicks):
        self.t += 1  # update time
        self.tot_sales[arm_pulled] += sales
        self.tot_clicks[arm_pulled] += clicks


class UCB(Learner):

    def __init__(self, env, prod, n_arms=4, c=1):
        super().__init__(n_arms)
        self.means = np.zeros(n_arms)  # to collect means
        self.widths = np.array([np.inf for _ in range(n_arms)])  # to collect upper confidence bounds (- mean)
        self.margins = env.global_margin[prod, :]  # I know the set of prices, not the conversion rates
        self.c = c

    # to select the arms with highest upper confidence bound
    def pull_cr(self):
        idx = np.array(self.means + self.widths, dtype=float)
        for i in range(4):
            if (idx[i] > 1) or (idx[i] == np.inf):
                idx[i] = 1
        return idx

    def update(self, arm_pulled, sales, clicks):
        super().update(arm_pulled, sales, clicks)
        self.means[arm_pulled] = self.tot_sales[arm_pulled]/self.tot_clicks[arm_pulled]  # update the mean of the arm we pulled
        for idx in range(self.n_arms):  # for all arms, update upper confidence bounds
            n = self.tot_clicks[idx]
            if n > 0:
                self.widths[idx] = self.c*np.sqrt(2 * np.log(self.t) / n)
            else:
                self.widths[idx] = np.inf


class Items_UCB_Learner:
    def __init__(self, env, n_items=5, n_arms=4, c=1):
        self.env = env
        self.learners = [UCB(env, i, n_arms, c) for i in range(n_items)]
        self.n_arms = n_arms
        self.n_items = n_items

    def pull_prices(self, env: Hyperparameters, print_message, n_users_pt=100):
        conv_rate = -1 * np.ones(shape=(5, 4))
        for i in range(5):
            conv_rate[i, :] = self.learners[i].pull_cr()
        prices = pull_prices(env=env, conv_rates=conv_rate, alpha=env.dir_params,n_buy=env.mepp,
                                      trans_prob=env.global_transition_prob, n_users_pt=n_users_pt,
                                      print_message=print_message)
        return prices

    def update(self, day):
        for i in range(self.n_items):
            self.learners[i].update(day.pulled_prices[i], clicks=day.individual_clicks[i], sales=day.individual_sales[i])
