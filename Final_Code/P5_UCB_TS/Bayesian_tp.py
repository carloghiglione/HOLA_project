import copy

import numpy as np
import sys
sys.path.insert(0, '..')
from P1_Base.MC_simulator import pull_prices
from Classes_5 import Hyperparameters, Day

class Bayesian_TP:
    def __init__(self, first: bool, lam: float):
        self.param = np.array([1,1])
        self.first = first
        self.lam = lam

    def update(self, n_display, n_clicks):
        self.param[0] += n_clicks
        self.param[1] += (n_display - n_clicks)

    def pull_value(self):
        value = np.random.beta(a=self.param[0], b=self.param[1], size=1)
        if (not self.first):
            if value > self.lam:
                value = copy.deepcopy(self.lam)
        return value

class Full_Bayesian_TP:
    def __init__(self, env: Hyperparameters):
        self.trans_order = env.trans_order
        learners = [[] for _ in range(5)]
        for i in range(5):
            for j in range(5):
                first = (self.trans_order[i, j] == 1)
                temp = Bayesian_TP(first=first, lam=env.lam)
                learners[i].append(copy.deepcopy(temp))
        self.learners = copy.deepcopy(learners)

    def update(self, day: Day):
        for i in range(5):
            for j in range(5):
                if self.trans_order[i, j] > 0:
                    self.learners[i][j].update(n_display=day.shows[i, j], n_clicks=day.cliks[i, j])

    def pull_prices(self, env: Hyperparameters, print_message, n_users_pt=100):
        trans_prob = np.zeros(shape=(5, 5))
        for i in range(5):
            for j in range(5):
                if self.trans_order[i, j] > 0:
                    trans_prob[i, j] = self.learners[i][j].pull_value()
        prices = pull_prices(env=env, conv_rates=env.global_conversion_rate, alpha=env.dir_params, n_buy=env.mepp,
                             trans_prob=trans_prob, n_users_pt=n_users_pt, print_message=print_message)
        return prices