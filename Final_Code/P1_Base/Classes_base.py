import numpy as np
import numpy.random as npr
import math as mt
import copy

#Fixed parameters:
#   •dirichlet parameters for each user type -> AVERAGE partition of users amongst products
#   •poisson parameters -> MEAN number of users for each type each day

class Hyperparameters:
    def __init__(self, transition_prob_listofmatrix, dir_params_listofvector, pois_param_vector, conversion_rate_listofmatrix, margin_matrix, mean_extra_purchases_per_product=2*np.ones(shape=(3, 5))):
        self.global_transition_prob = transition_prob_listofmatrix  # transition_prob[i,j] = prob that j is selected given i as primal, this econdes both selection probability and probability to see j
        self.dir_params = dir_params_listofvector  # vector with dirichlet parameters
        self.pois_param = pois_param_vector
        self.global_conversion_rate = conversion_rate_listofmatrix  # conversion_rate[i,j] = conversion rate of product i for price j
        self.global_margin = margin_matrix  # margin[i,j] = margin of item i for price j
        self.mepp = mean_extra_purchases_per_product


class Daily_Website:
    def __init__(self, env: Hyperparameters, pulled_prices):
        self.transition_prob = env.global_transition_prob
        self.alphas = self.sample_user_partitions(env.dir_params)
        self.n_users = self.sample_n_users(env.pois_param)
        self.price = pulled_prices.astype(int)
        self.n_estimated_types = pulled_prices.shape[0]
        self.conversion_rates = self.select_conversion_rates(env.global_conversion_rate, pulled_prices)
        self.margin = self.select_margins(env.global_margin, pulled_prices)
        self.env = env

    def sample_user_partitions(self, params):
        alphas = []
        for i in range(3):
            alphas.append(npr.dirichlet(alpha=params[i], size=1)[0])
        return alphas

    def sample_n_users(self, params):
        n_users = []
        for i in range(3):
            n_users.append(int(params[i]*0.95 + npr.poisson(lam=params[i]*0.05, size=1)))
        return n_users

    def select_conversion_rates(self, conv_rates, prices):
        ret = np.ndarray(shape=(3, 5), dtype=float)
        for i in range(3):
            for j in range(5):
                ret[i, j] = conv_rates[i][j, prices[j]]
        return ret

    def select_margins(self, margins, prices):
        ret = np.zeros(5, dtype=float)
        for j in range(5):
            ret[j] = margins[j, prices[j]]
        return ret

    def get_users_per_product_and_type(self):
        users_pp = np.ndarray(shape=(3, 5), dtype=int)
        for t in range(3):
            users = []
            total = 0
            for i in range(5):
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
            users_pp[t, :] = np.array(users, dtype=int)
        return users_pp


class User:
    def __init__(self, website: Daily_Website, starting_product, u_type):
        self.website = website  # environment is the specific day website
        self.u_type = u_type
        self.mepp = website.env.mepp
        self.starting_product = starting_product
        self.products = np.zeros(5, dtype=int)  # 1 if product has been bought, 0 if not
        self.clicked = np.zeros(5, dtype=int)  # 1 if product has been clicked, 0 if not
        self.cart = np.zeros(5, dtype=int)  # n* elements per product
        self.dynamic_transition_prob = copy.deepcopy(website.transition_prob[self.u_type])

    def new_primary(self, primary):
        self.clicked[primary] = 1
        for i in range(5):
            # now that it is shown as primal, it can never be selected by other products
            self.dynamic_transition_prob[i, primary] = 0
        buy = npr.binomial(n=1, size=1, p=self.website.conversion_rates[self.u_type, primary])

        if buy:
            self.products[primary] = 1
            how_much = npr.poisson(size=1, lam=self.mepp[self.u_type, primary])[0]+1  # +1 since we know the user buys
            self.cart[primary] = how_much
            for j in range(5):
                click = npr.binomial(n=1, size=1, p=self.dynamic_transition_prob[primary, j])
                if click:
                    self.new_primary(j)  # depth first approach

    def simulate(self):
        self.new_primary(self.starting_product)

    def checkout(self) -> float:
        singular_margin = 0
        for i in range(5):
            singular_margin = singular_margin + self.cart[i]*self.website.margin[i]
        return singular_margin


class Day:
    def __init__(self, g_web: Hyperparameters, prices):
        self.pulled_prices = prices.astype(int)
        self.env = g_web
        self.mepp = g_web.mepp
        self.profit = 0.
        self.website = Daily_Website(g_web, self.pulled_prices)
        self.n_users = self.website.get_users_per_product_and_type()
        # n items sold per type of product
        self.items_sold = [0 for _ in range(5)]
        # n costumers that bought type of product, regardless of the amount
        self.individual_sales = [0 for _ in range(5)]
        # n costumers that clicked type of product, regardless of if they bought
        self.individual_clicks = [0 for _ in range(5)]

    def run_simulation(self):
        for t in range(3):
            for p in range(5):
                for j in range(self.n_users[t, p]):
                    user = User(self.website, p, t)
                    user.simulate()
                    self.profit = self.profit + user.checkout()
                    # self.items_sold = self.items_sold + user.cart elementwise
                    self.items_sold = [sum(x) for x in zip(self.items_sold, user.cart)]
                    self.individual_sales = [sum(x) for x in zip(self.individual_sales, user.products)]
                    self.individual_clicks = [sum(x) for x in zip(self.individual_clicks, user.clicked)]

    def run_clairvoyant_simulation(self, best_prices) -> float:
        best_prices = best_prices.astype(int)
        best_website = Daily_Website(self.env, best_prices)
        profit = 0
        for t in range(3):
            for p in range(5):
                for j in range(self.n_users[t, p]):
                    user = User(best_website, p, t)
                    user.simulate()
                    profit = profit + user.checkout()
        return profit
