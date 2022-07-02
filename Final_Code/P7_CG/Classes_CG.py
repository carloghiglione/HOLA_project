import numpy as np
import numpy.random as npr
import math as mt
import copy

# Fixed parameters:
#   •dirichlet parameters for each user type -> AVERAGE partition of users amongst products
#   •poisson parameters -> MEAN number of users for each type each day

class Hyperparameters:
    def __init__(self, transition_prob_listofmatrix, dir_params_listofvector, pois_param_vector, conversion_rate_listofmatrix, margin_matrix, feat_ass: np.array, mean_extra_purchases_per_product=2*np.ones(shape=(3, 5))):
        self.global_transition_prob = transition_prob_listofmatrix  # transition_prob[i,j] = prob that j is selected given i as primal, this econdes both selection probability and probability to see j
        self.dir_params = dir_params_listofvector  # vector with dirichlet parameters
        self.pois_param = pois_param_vector
        self.global_conversion_rate = conversion_rate_listofmatrix  # conversion_rate[i,j] = conversion rate of product i for price j
        self.global_margin = margin_matrix  # margin[i,j] = margin of item i for price j
        self.mepp = mean_extra_purchases_per_product
        self.feature_associator = feat_ass


class Daily_Website:
    def __init__(self, env: Hyperparameters, pulled_prices):
        self.env = env
        self.transition_prob = env.global_transition_prob
        self.alphas = self.sample_user_partitions(env.dir_params)
        self.n_users = self.sample_n_users(env.pois_param)
        self.price = pulled_prices
        self.conversion_rates = self.select_conversion_rates(env.global_conversion_rate, pulled_prices)
        self.margin = self.select_margins(env.global_margin, pulled_prices)

    def sample_user_partitions(self, params):
        alphas = []
        for i in range(3):
            alphas.append(npr.dirichlet(alpha=params[i], size=1)[0])
        return alphas

    def sample_n_users(self, params):
        n_users = np.zeros(shape=(2, 2), dtype=int)
        for f1 in range(2):
            for f2 in range(2):
                n_users[f1, f2] = int(params[f1][f2]*0.95 + npr.poisson(lam=params[f1][f2]*0.05, size=1))
        return n_users

    def select_conversion_rates(self, conv_rates, prices):
        ret = np.zeros(shape=(2, 2, 5), dtype=float)
        for f1 in range(2):
            for f2 in range(2):
                for j in range(5):
                    true_type = self.env.feature_associator[f1][f2]
                    ret[f1, f2, j] = conv_rates[true_type][j, prices[f1, f2, j]]
        return ret

    def select_margins(self, margins, prices):
        ret = np.zeros(shape=(2, 2, 5), dtype=float)
        for f1 in range(2):
            for f2 in range(2):
                for j in range(5):
                    ret[f1, f2, j] = margins[j, prices[f1, f2, j]]
        return ret

    def get_users_per_product_and_type(self):
        users_pp = np.ndarray(shape=(2, 2, 5), dtype=int)
        for f1 in range(2):
            for f2 in range(2):
                t = self.env.feature_associator[f1][f2]
                users = []
                total = 0
                for i in range(5):
                    n_dirty = self.n_users[f1, f2] * self.alphas[t][i + 1]
                    if (n_dirty % 1 > 0.5) and (total + mt.ceil(n_dirty) <= self.n_users[f1, f2]):
                        users.append(mt.ceil(n_dirty))
                        total = total + mt.ceil(n_dirty)
                    elif total + mt.floor(n_dirty) <= self.n_users[f1, f2]:
                        users.append(mt.floor(n_dirty))
                        total = total + mt.floor(n_dirty)
                    else:
                        print(str('\n'+"ERROR IN USER EXTRACTION"+'\n'))
                users_pp[f1, f2, :] = np.array(users, dtype=int)
        return users_pp


class User:
    def __init__(self, website: Daily_Website, starting_product, u_type, feats):
        self.website = website  # environment is the specific day website
        self.u_type = u_type
        self.features = feats
        self.mepp = website.env.mepp
        self.starting_product = starting_product
        self.products = [0 for _ in range(5)]  # 1 if product has been bought, 0 if not
        self.clicked = [0 for _ in range(5)]  # 1 if product has been clicked, 0 if not
        self.cart = [0 for _ in range(5)]  # n* elements per product
        self.dynamic_transition_prob = copy.deepcopy(website.transition_prob[self.u_type])

    def new_primary(self, primary):
        self.clicked[primary] = 1
        for i in range(5):
            # now that it is shown as primal, it can never be selected by other products
            self.dynamic_transition_prob[i, primary] = 0
        buy = npr.binomial(n=1, size=1, p=self.website.conversion_rates[self.features[0], self.features[1], primary])

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
            singular_margin = singular_margin + self.cart[i]*self.website.margin[self.features[0], self.features[1], i]
        return singular_margin


class Day:
    def __init__(self, g_web: Hyperparameters, prices):
        self.pulled_prices = prices
        self.env = g_web
        self.mepp = g_web.mepp
        self.profit = 0.
        self.website = Daily_Website(g_web, self.pulled_prices)
        self.n_users = self.website.get_users_per_product_and_type()
        # n items sold per type of product
        self.items_sold = np.zeros(shape=(2, 2, 5), dtype=int)
        # n costumers that bought type of product, regardless of the amount
        self.individual_sales = np.zeros(shape=(2, 2, 5), dtype=int)
        # n costumers that clicked type of product, regardless of if they bought
        self.individual_clicks = np.zeros(shape=(2, 2, 5), dtype=int)

    def run_simulation(self):
        for f1 in range(2):
            for f2 in range(2):
                for p in range(5):
                    for j in range(self.n_users[f1, f2, p]):
                        true_type = self.env.feature_associator[f1, f2]
                        user = User(website=self.website, starting_product=p, u_type=true_type, feats=(f1, f2))
                        user.simulate()
                        self.profit = self.profit + user.checkout()
                        # self.items_sold = self.items_sold + user.cart elementwise
                        self.items_sold[f1, f2, :] = self.items_sold[f1, f2, :] + np.array(user.cart, dtype=int)
                        self.individual_sales[f1, f2, :] = self.individual_sales[f1, f2, :] + np.array(user.products, dtype=int)
                        self.individual_clicks[f1, f2, :] = self.individual_clicks[f1, f2, :] + np.array(user.clicked, dtype=int)

    def run_clairvoyant_simulation(self, best_prices) -> float:
        best_website = Daily_Website(self.env, best_prices)
        profit = 0

        for f1 in range(2):
            for f2 in range(2):
                for p in range(5):
                    for j in range(self.n_users[f1, f2, p]):
                        true_type = self.env.feature_associator[f1, f2]
                        user = User(website=best_website, starting_product=p, u_type=true_type, feats=(f1, f2))
                        user.simulate()
                        profit = profit + user.checkout()
        return profit
