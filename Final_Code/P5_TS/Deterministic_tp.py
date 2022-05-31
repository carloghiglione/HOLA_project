import copy
import numpy as np
from P1_Base.Price_puller import pull_prices
from Classes_5 import Hyperparameters, Day


class Deterministic_TP:
    def __init__(self):
        self.n_clicks = 0
        self.n_display = 0

    def update(self, n_display, n_clicks):
        self.n_clicks += n_clicks
        self.n_display += n_display

    def pull_value(self):
        if self.n_display == 0:
            return 0
        else:
            value = float(self.n_clicks/self.n_display)
            return value


class Full_Deterministic_TP:
    def __init__(self, env: Hyperparameters):
        self.trans_order = env.trans_order
        learners = [[] for _ in range(5)]
        for i in range(5):
            for j in range(5):
                temp = Deterministic_TP()
                learners[i].append(copy.deepcopy(temp))
        self.learners = copy.deepcopy(learners)

    def update(self, day: Day):
        for i in range(5):
            for j in range(5):
                if self.trans_order[i, j] > 0:
                    self.learners[i][j].update(n_display=day.shows[i, j], n_clicks=day.cliks[i, j])

    def pull_prices(self, env: Hyperparameters, print_message):
        trans_prob = np.zeros(shape=(5, 5), dtype=float)
        for i in range(5):
            for j in range(5):
                if self.trans_order[i, j] > 0:
                    trans_prob[i, j] = self.learners[i][j].pull_value()
        prices = pull_prices(env=env, conv_rates=env.global_conversion_rate, alpha=env.dir_params, n_buy=env.mepp,
                             trans_prob=trans_prob, print_message=print_message)
        return prices
