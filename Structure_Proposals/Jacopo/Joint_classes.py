import numpy as np
import numpy.random as npr
import math as mt
import copy

class Environment:
    def __init__(self, dir_par, pois_par, n_classes = 3, n_prod = 5):
        self.dir_par = dir_par
        self.n_classes = n_classes
        self.pois_par = pois_par
        self.n_prod = n_prod

    def alphas(self, cl):
        return npr.dirichlet(alpha = self.dir_par[cl], size = 1)

    def n_users(self,cl):
        return npr.poisson(lam=self.pois_par[cl], size = 1)

    def return_users(self):
        first_pages = []
        classes = []
        cl = np.array(range(self.n_prod+1))
        for i in range(self.n_classes):
            for j in range(self.n_users()):
                classes.append(i)
                first_pages.append(npr.choice(a = cl, size = 1, p = self.alphas()))
        return classes, first_pages

class WebSite:
    def __init__(self, connection_matrix, prices, choiche):
        self.con_matrix = connection_matrix
        self.base_prices = prices
        self.current_prices = self.base_prices[choiche]