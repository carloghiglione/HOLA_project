from Classes import *
import numpy as np

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

    def __init__(self, env : Hyperparameters, prod, n_arms=4, c=1):
        super().__init__(n_arms)
        self.means = np.zeros(n_arms)  # to collect means
        self.widths = np.array([np.inf for _ in range(n_arms)])  # to collect upper confidence bounds (- mean)
        self.margins = env.global_margin[prod, :]  # I know the set of prices, not the conversion rates
        self.c = c

    # to select the arms with highest upper confidence bound
    def pull_arm(self):
        idx = np.argmax(
            (self.means + self.widths) * self.margins)  # I multiply everything by the prices, then return the max
        return int(idx)

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
    def __init__(self, env: Hyperparameters, n_items=5, n_arms=4, c=1):
        self.env = env
        self.learners = [UCB(env, i, n_arms, c) for i in range(n_items)]
        self.n_arms = n_arms
        self.n_items = n_items

    def pull_prices(self):
        idx = -1*np.ones(self.n_items, dtype=int)
        for i in range(self.n_items):
            idx[i] = self.learners[i].pull_arm()
        return idx

    def update(self, day: Day):
        for i in range(self.n_items):
            self.learners[i].update(day.pulled_prices[i], clicks= day.individual_clicks[i], sales= day.individual_sales[i])
