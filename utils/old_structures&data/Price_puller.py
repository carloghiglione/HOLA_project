import copy
import numpy as np
import sys
from copy import deepcopy as cdc
from P1_Base.Classes_base import Hyperparameters, Daily_Website

# SIMULATORE CON:
# -conversion rates
# -alpha
# -n.ro acquisti
# -transition prob.


def profit_puller(prices, env: Hyperparameters, n_users_pt, tr_prob) -> float:

    env_daily = Daily_Website(env, cdc(prices))
    env_daily.n_users = [n_users_pt, n_users_pt, n_users_pt]
    env_daily.alphas = np.array(env.dir_params, dtype=float)/np.sum(env.dir_params)

    # tran_prob = (MC_daily.transition_prob[0]+MC_daily.transition_prob[1]+MC_daily.transition_prob[2])/3
    tran_prob = tr_prob
    alphas = (env_daily.alphas[0] + env_daily.alphas[1] + env_daily.alphas[2]) / 3
    conv_rate = np.mean(env_daily.conversion_rates, axis=0)

    connectivity = np.zeros(shape=(5, 2), dtype=int)
    for i in range(5):
        connectivity[i, :] = np.array(np.where(tran_prob[i, :] > 0))

    pur_prob = np.zeros(5, dtype=float)

    all_prods = np.array([0, 1, 2, 3, 4])

    for p1 in range(5):
        visited = np.array([p1])
        to_visit = np.delete(copy.deepcopy(all_prods), visited)

        click_in_chain = np.zeros(5, dtype=float)

        click_in_chain[p1] = conv_rate[p1]
        prob_per_p1 = np.zeros(5, dtype=float)

        for p2 in np.intersect1d(connectivity[p1], to_visit):
            visited = np.array([p1, p2])
            to_visit = np.delete(copy.deepcopy(all_prods), visited)

            click_in_chain[p2] = conv_rate[p2]*tran_prob[p1, p2]*click_in_chain[p1]*(1-prob_per_p1[p2])
            prob_per_p1[p2] += click_in_chain[p2]

            for p3 in np.intersect1d(connectivity[p2], to_visit):
                visited = np.array([p1, p2, p3])
                to_visit = np.delete(copy.deepcopy(all_prods), visited)

                click_in_chain[p3] = conv_rate[p3]*tran_prob[p2, p3]*click_in_chain[p2]*(1-prob_per_p1[p3])
                prob_per_p1[p3] += click_in_chain[p3]

                for p4 in np.intersect1d(connectivity[p3], to_visit):
                    visited = np.array([p1, p2, p3, p4])
                    to_visit = np.delete(copy.deepcopy(all_prods), visited)

                    click_in_chain[p4] = conv_rate[p4]*tran_prob[p3, p4]*click_in_chain[p3]*(1-prob_per_p1[p4])
                    prob_per_p1[p4] += click_in_chain[p4]

                    for p5 in np.intersect1d(connectivity[p4], to_visit):

                        prob_per_p1[p5] += conv_rate[p5]*tran_prob[p4, p5]*click_in_chain[p4]*(1 - prob_per_p1[p5])

        prob_per_p1[p1] = conv_rate[p1]
        pur_prob += prob_per_p1*alphas[p1+1]
    profit = float(np.sum(pur_prob*env_daily.margin*(1.0 + env.mepp)))

    return profit


def pull_prices(env: Hyperparameters, conv_rates, alpha, n_buy, trans_prob, n_users_pt=100, print_message="Simulating")\
        -> np.array:
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
        conv_rate = [conv_rate for _ in range(3)]
    if len(tran_prob) != 3:
        tran_prob = [tran_prob for _ in range(3)]
    if len(alpha) != 3:
        alpha = [alpha for _ in range(3)]

    env = Hyperparameters(tran_prob, alpha, envv.pois_param, conv_rate, envv.global_margin, n_buy)

    tr_prob = (tran_prob[0]+tran_prob[1]+tran_prob[2])/3

    count = 0
    cc = 4**5
    prices = [-1*np.ones(5) for _ in range(cc)]
    profits = np.zeros(cc, dtype=int)

    sim_prices = np.zeros(5, dtype=int)

    for p1 in range(4):
        sim_prices[0] = p1
        for p2 in range(4):
            sim_prices[1] = p2
            for p3 in range(4):
                sim_prices[2] = p3
                for p4 in range(4):
                    sim_prices[3] = p4
                    for p5 in range(4):
                        sim_prices[4] = p5
                        profits[count] = profit_puller(sim_prices, cdc(env), n_users_pt, tr_prob)
                        prices[count] = cdc(sim_prices)

                        count += 1
                    sys.stdout.write('\r' + print_message + str(", pulling prices: ") + f'{count * 100 / cc} %')

    profits = np.array(profits, dtype=float)
    best = np.argmax(profits)
    return prices[best]
