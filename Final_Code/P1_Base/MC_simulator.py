import numpy as np
import sys
from copy import deepcopy as cdc
from P1_Base.Classes_base import *

#SIMULATORE CON:
#-conversion rates
#-alpha
#-n.ro acquisti
#-transition prob.

def single_MC_simulator(prices, MC_env: Hyperparameters, n_users_pt) -> float:

    MC_daily = Daily_Website(MC_env, cdc(prices))
    MC_daily.n_users = [n_users_pt, n_users_pt, n_users_pt]
    MC_daily.alphas = np.array(MC_env.dir_params, dtype=float)/np.sum(MC_env.dir_params)

    MC_day = Day(cdc(MC_env), cdc(prices))
    MC_day.website = cdc(MC_daily)
    MC_day.n_users = MC_day.website.get_users_per_product_and_type()

    MC_day.run_simulation()

    return MC_day.profit


def pull_prices(env: Hyperparameters, conv_rates, alpha, n_buy, trans_prob, n_users_pt=100, print_message="Simulating") -> np.array:
    prices = []
    profits = []
    conv_rate = cdc(conv_rates)
    tran_prob = cdc(trans_prob)
    envv = cdc(env)
    if len(conv_rate) != 3:  # SE SONO PASSATI GLI STIMATORI E NON QUELLI VERI
        for i in range(5):
            for j in range(4):
                if (conv_rate[i][j] > 1) or (np.isinf(conv_rate[i][j])):
                    conv_rate[i][j] = 1

    if len(tran_prob) != 3:  # SE SONO PASSATI GLI STIMATORI E NON QUELLI VERI
        for i in range(5):
            for j in range(5):
                if (tran_prob[i][j] > 1) or (np.isinf(tran_prob[i][j])):
                    tran_prob[i][j] = 1

    if len(conv_rate) != 3:                          # if I am in the case with one class only
        conv_rate = [conv_rate for i in range(3)]
    if len(tran_prob) != 3:
        trans_prob = [tran_prob for i in range(3)]
    if len(alpha) != 3:
        alpha = [alpha for i in range(3)]

    MC_env = Hyperparameters(tran_prob, alpha, envv.pois_param, conv_rate, envv.global_margin, n_buy)

    count = 1
    cc = 4**5
    for p1 in range(4):
        for p2 in range(4):
            for p3 in range(4):
                for p4 in range(4):
                    for p5 in range(4):
                        sim_prices = np.array([p1, p2, p3, p4, p5])
                        profits.append(single_MC_simulator(sim_prices, cdc(MC_env), n_users_pt))
                        prices.append(sim_prices)
                        sys.stdout.write('\r' + print_message + str(", pulling prices: ") + f'{count * 100 / cc} %')
                        count += 1

    profits = np.array(profits, dtype=float)
    best = np.argmax(profits)
    return prices[best]
