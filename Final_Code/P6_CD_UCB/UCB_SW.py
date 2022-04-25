import numpy as np

class Learner_SW:

    def __init__(self, n_arms, win):
        self.t = 0
        self.n_arms = n_arms
        self.win = win
        self.ages = [[] for i in range(n_arms)]
        self.win_clicks = [[] for i in range(n_arms)]
        self.win_sales = [[] for i in range(n_arms)]

    # function to reset to initial condition
    def reset(self):
        self.__init__(self.n_arms, self.win)

    # perform one step of algo to select the arm that I pull, specific of the algorithm I use
    def act(self):
        pass

    def plus_one_age(self):
        for i in range(self.n_arms):
            if len(self.ages[i]) == 0:
                break
            for j in range(len(self.ages[i])):
                self.ages[i][j] += 1
            if self.ages[i][0] > self.win:
                self.ages[i].pop(0)
                self.win_sales[i].pop(0)
                self.win_clicks[i].pop(0)

    # collect output of algo and append the reward
    def update(self, arm_pulled: int, sales: int, clicks: int):
        self.t += 1  # update time
        self.plus_one_age()
        self.ages[arm_pulled].append(0)
        self.win_sales[arm_pulled].append(sales)
        self.win_clicks[arm_pulled].append(clicks)

class UCB_SW(Learner_SW):

    def __init__(self, env, prod, win, n_arms=4, c=1):
        super().__init__(n_arms, win)
        self.means = np.zeros(n_arms)  # to collect means
        self.widths = np.array([np.inf for _ in range(n_arms)])  # to collect upper confidence bounds (- mean)
        self.margins = env.global_margin[prod, :]  # I know the set of prices, not the conversion rates
        self.c = c
        self.win = win

    # to select the arms with highest upper confidence bound
    def pull_arm(self):
        idx = np.argmax(
            (self.means + self.widths) * self.margins)  # I multiply everything by the prices, then return the max
        return int(idx)

    def update(self, arm_pulled, sales, clicks):
        super().update(arm_pulled, sales, clicks)
          # update the mean of the arm we pulled
        for idx in range(self.n_arms):  # for all arms, update upper confidence bounds
            n = len(self.ages[idx])
            if n > 0:
                self.means[idx] = np.sum(self.win_sales[idx]) / np.sum(self.win_clicks[idx])
                self.widths[idx] = self.c*np.sqrt(2 * np.log(np.min((self.t, self.win))) / np.sum(self.win_clicks[idx]))
            else:
                self.means[idx] = 0
                self.widths[idx] = np.inf


class Items_UCB_Learner_SW:
    def __init__(self, env, win, n_items=5, n_arms=4, c=1):
        self.env = env
        self.learners = [UCB_SW(env, i, win ,n_arms, c) for i in range(n_items)]
        self.n_arms = n_arms
        self.n_items = n_items

    def pull_prices(self):
        idx = -1*np.ones(self.n_items, dtype=int)
        for i in range(self.n_items):
            idx[i] = self.learners[i].pull_arm()
        return idx

    def update(self, day):
        for i in range(self.n_items):
            self.learners[i].update(day.pulled_prices[i], clicks=day.individual_clicks[i], sales=day.individual_sales[i])
