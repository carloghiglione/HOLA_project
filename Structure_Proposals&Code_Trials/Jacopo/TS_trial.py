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


class TS(Learner):

    def __init__(self,  env : Hyperparameters, prod, n_arms=4):
        super().__init__(n_arms)  # usa il costruttore di learner (super-class)
        self.env = env
        self.beta_parameters = np.ones((n_arms, 2))  # variable to store parameters of beta distributions for each arm
        self.margins = env.global_margin[prod,:]

    # to select the arm to  pull
    # sample from betas of all the arms and find arm that sampled the maximum value
    def pull_arm(self):
        idx = np.argmax(np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1])*self.margins)
        return idx

    def update(self, pulled_arm, sales, clicks):
        super().update(pulled_arm, sales, clicks)
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + sales  # update first param of beta
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[
                                                  pulled_arm, 1] + clicks - sales  # update second param of beta


class Item_TS_Learner:

    def __init__(self, env: Hyperparameters, n_items=5, n_arms=4):
        self.env = env
        self.learners = [TS(env, i, n_arms) for i in range(n_items)]
        self.n_arms = n_arms
        self.n_items = n_items

    def pull_prices(self):
        idx = -1*np.ones(self.n_items)
        for i in range(self.n_items):
            idx[i] = self.learners[i].pull_arm()
        return idx

    def update(self, day: Day):
        for i in range(self.n_items):
            self.learners[i].update(pulled_arm=day.pulled_prices[i],
                                    sales=day.individual_sales[i], clicks=day.individual_clicks[i])
