import numpy as np
import numpy.random as npr
import math as mt
import copy

from Classes import Hyperparameters

# TODO DEFINE FEATURE-TYPE MECHANISM IN DAY


class Daily_Website_types:
    def __init__(self, env : Hyperparameters, pulled_prices : np.ndarray):
        self.transition_prob = env.global_transition_prob
        self.alphas = self.sample_user_partitions(env.dir_params)
        self.n_users = self.sample_n_users(env.pois_param)
        self.price = pulled_prices
        self.n_estimated_types = pulled_prices.shape[0]
        self.conversion_rates = self.select_conversion_rates(env.global_conversion_rate, pulled_prices)
        self.margin = self.select_margins(env.global_margin, pulled_prices)

    def sample_user_partitions(self, params):
        alphas = []
        for i in range(3):
            alphas.append(npr.dirichlet(alpha=params[i], size=1))
        return alphas

    def sample_n_users(self, params):
        n_users = []
        for i in range(3):
            n_users.append(npr.poisson(lam = params[i], size=1))
        return n_users

    def select_conversion_rates(self, conv_rates, prices):
        ret = np.ndarray(shape=(3, 5))
        for i in range(3):
            for j in range(5):
                for t in range(self.n_estimated_types):
                    ret[i, t, j] = conv_rates[i][j,prices[t,j]]
        return ret

    def select_margins(self, margins, prices):
        ret = np.ndarray(shape=(self.n_estimated_types, 5))
        for i in range(self.n_estimated_types):
            for j in range(5):
                ret[i, j] = margins[j, prices[i, j]]
        return ret

    def get_users_per_product_and_type(self):
        users_pp = np.ndarray(shape=(3, 6))
        for t in range(3):
            users = []
            total = 0
            for i in range(4):
                n_dirty = self.n_users[t] * self.alphas[t][i+1]
                if (n_dirty % 1 > 0.5) and (total + mt.ceil(n_dirty) <= self.n_users[t]):
                    users.append(mt.ceil(n_dirty))
                    total = total + mt.ceil(n_dirty)
                elif total + mt.floor(n_dirty) <= self.n_users[t]:
                    users.append(mt.floor(n_dirty))
                    total = total + mt.floor(n_dirty)
                else:
                    for j in range(25):
                        print("ERROR IN USER EXTRACTION")
            users.append(self.n_users[t] - total)
            users_pp[t, :] = np.array(users)
        return users_pp

class User_types:  # CHECK .checkout() WHEN DIFFERENT TYPES ARE STUDIED
    def __init__(self, website: Daily_Website_types, starting_product, u_type, feature1, feature2,
                 mean_purchases_per_product=2*np.ones(shape=5)):
        self.website = website  # environment is the specific day website
        self.u_type = u_type
        self.est_type = self.estimate_type(feature1, feature2)
        self.mean_purchases_per_product = mean_purchases_per_product
        self.starting_product = starting_product
        self.products = [0 for i in range(5)]  # 1 if product has been bought, 0 if not
        self.clicked = [0 for i in range(5)]  # 1 if product has been clicked, 0 if not
        self.cart = [0 for i in range(5)]  # n* elements per product
        self.dynamic_transition_prob = copy.deepcopy(website.transition_prob[self.u_type])

    def estimate_type(self, feature1, feature2):
        dosomething = 0
        return 1

    def new_primary(self, primary):
        self.clicked[primary] = 1
        for i in range(5):
            # now that it is shown as primal, it can never be selected by other products
            self.dynamic_transition_prob[i, primary] = 0
        buy = npr.binomial(n=1, size=1, p=self.website.conversion_rates[self.u_type, self.est_type, primary])
        if buy:
            self.products[primary] = 1
            how_much = npr.poisson(size=1, lam=self.mean_purchases_per_product[primary])+1
            self.cart[primary] = how_much
            for j in range(5):
                click = npr.binomial(n=1, size=1, p=self.dynamic_transition_prob[primary, j])
                if click:
                    self.new_primary(j)  # depth first approach

    def simulate(self):
        self.new_primary(self.starting_product)

    def checkout(self):
        singular_margin = 0
        for i in range(5):
            singular_margin = singular_margin + self.cart[i] * self.website.margin[self.est_type, i]
        return singular_margin

class Day_types:
    def __init__(self, g_web: Hyperparameters, prices):
        self.pulled_prices = prices
        self.profit = 0
        self.website = Daily_Website_types(g_web, self.pulled_prices)
        self.n_users = self.website.get_users_per_product_and_type()
        self.items_sold = [0 for i in range(5)]  # n items sold per type of product
        self.individual_sales = [0 for i in range(5)]
        self.individual_clicks = [0 for i in range(5)]

    def run_simulation(self):
        for t in range(3):
            for p in range(5):
                for j in range(self.n_users[t, p]):
                    f1 = 1  # TODO SETTALE
                    f2 = 1
                    user = User_types(self.website, p, t, f1, f2)
                    user.simulate()
                    self.profit = self.profit + user.checkout()
                    self.items_sold = [sum(x) for x in zip(self.items_sold, user.cart)]
                    self.individual_sales = [sum(x) for x in zip(self.individual_sales, user.products)]
                    self.individual_clicks = [sum(x) for x in zip(self.individual_clicks, user.clicked)]