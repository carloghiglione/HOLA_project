import numpy as np
import numpy.random as npr
import math as mt
import copy

#TODO DEFINE CLASS BETWEEN USERS
#TODO DEFINE MODEL FOR HOW MANY SALES

class Global_Website:
    def __init__(self, transition_prob_matrix, dir_params_vector, n_users_int, conversion_rate_matrix, margin_matrix):
        self.global_transition_prob = transition_prob_matrix #transition_prob[i,j] = prob that j is selected given i as primal, this econdes both selection probability and probability to see j
        self.dir_params = dir_params_vector #vector with dirichlet parameters
        self.global_n_users = n_users_int
        self.global_conversion_rate = conversion_rate_matrix#conversion_rate[i,j] = conversion rate of product i for price j
        self.global_margin = margin_matrix#margin[i,j] = margin of item i for product j
        
    def sample_user_partition(self):
        return npr.dirichlet(alpha = self.dir_params, size = 1)

class Day_Website:
    def __init__(self, g_web, pulled_prices):
        self.transition_prob = g_web.global_transition_prob
        self.alpha = g_web.sample_user_partition()
        self.n_users = g_web.global_n_users
        self.price = pulled_prices
        self.conversion_rate = [g_web.global_conversion_rate[i,pulled_prices[i]] for i in range(5)]
        self.margin = [g_web.global_margin[i,pulled_prices[i]] for i in range(5)]

    def get_users_per_product(self):
        users = []
        total = 0
        for i in range(4):
            n_dirty = self.n_users * self.alpha[i+1]
            if (n_dirty % 1 > 0.5) and (total + mt.ceil(n_dirty) <= self.n_users):
                users.append(mt.ceil(n_dirty))
                total = total + mt.ceil(n_dirty)
            elif total + mt.floor(n_dirty) <= self.n_users:
                users.append(mt.floor(n_dirty))
                total = total + mt.floor(n_dirty)
            else:
                for j in range(25):
                    print("ERROR IN USER EXTRACTION")
        users.append(self.n_users - total)
        return users

class User:
    def __init__ (self, environment, starting_product):
        self.environment = environment#environment is the specific day website
        self.starting_product = starting_product
        self.products = [0 for i in range(5)]#1 if if product has been bought, 0 if not
        self.clicked = [0 for i in range(5)]#1 if if product has been clicked, 0 if not
        self.cart = [0 for i in range(5)]#n* elements per product
        self.dynamic_transition_prob = copy.deepcopy(environment.transition_prob)

    def new_primary(self, primary):

        self.clicked[primary] = 1
        for i in range(5):
            self.dynamic_transition_prob[i, primary] = 0#now that it is shown as primal, it can never be selected by other products
        buy = npr.binomial(n = 1, size = 1, p = self.environment.conversion_rate[primary])

        if buy:
            self.products[primary] = 1
            how_much = npr.poisson(size = 1, lam = 2) + 1#+1 since we know the user buys
            self.cart[primary] = how_much
            for i in range(5):
                click = npr.binomial(n = 1, size = 1, p = self.dynamic_transition_prob[primary, i])
                if click:
                    self.new_primary(i) #depth first approach

    def simulate(self):
        self.new_primary(self.starting_product)

    def checkout(self):
        singular_margin = 0
        for i in range(5):
            singular_margin = singular_margin + self.cart[i]*self.environment.margin[i]
        return singular_margin

class Day:
    def __init__ (self, g_web, prices):
        self.pulled_prices = prices
        self.profit = 0
        self.website = Day_Website(g_web, self.pulled_prices)
        self.users = self.website.get_users_per_product()
        self.items_sold = [0 for i in range(5)]#n items sold per type of product
        self.individual_sales = [0 for i in range(5)]#n costumers that bought type of product, regardless of the amount
        self.individual_clicks = [0 for i in range(5)]#n costumers that clicked type of product, regardless of if they bought


    def run_simulation(self):
        for i in range(5):
            for j in range(self.users[i]):
                user = User(self.website, i)
                user.simulate()
                self.profit = self.profit + user.checkout()
                self.items_sold = [sum(x) for x in zip(self.items_sold, user.cart)]# self.items_sold = self.items_sold + user.cart elementwise
                self.individual_sales = [sum(x) for x in zip(self.individual_sales, user.products)]
                self.individual_clicks = [sum(x) for x in zip(self.individual_clicks, user.clicked)]



for t in range(horizon_time):
    for i in range(5):
        day_prices = base_prices
        day_prices[i]++
        d = Day(environment, day_prices)
        d.run_simulation()
        decision_param = dosomething
        total_margin += d.profit()
        day_prices = getprices(decision_param)