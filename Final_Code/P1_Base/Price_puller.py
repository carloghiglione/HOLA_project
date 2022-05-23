import copy
import numpy as np
import sys
from copy import deepcopy as cdc
from P1_Base.Classes_base import Hyperparameters

# SIMULATORE CON:
# -conversion rates
# -alpha
# -n.ro acquisti
# -transition prob.


def profit_puller(prices, conv_rate_full, margins_full, tran_prob, alphas, mepp) -> float:

    conv_rate = np.zeros(shape=5, dtype=float)
    margin = np.zeros(shape=5, dtype=float)
    connectivity = np.zeros(shape=(5, 2), dtype=int)

    for j in range(5):
        conv_rate[j] = conv_rate_full[j, prices[j]]
        margin[j] = margins_full[j, prices[j]]
        connectivity[j, :] = np.array(np.where(tran_prob[j, :] > 0))

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
        pur_prob += prob_per_p1*(alphas[p1+1])
    profit = float(np.sum(pur_prob*margin*(1.0 + mepp)))

    return profit


def pull_prices(env: Hyperparameters, conv_rates, alpha, n_buy, trans_prob, print_message="Simulating") -> np.array:
    conv_rate = cdc(conv_rates)
    tran_prob = cdc(trans_prob)
    if len(conv_rate) != 3:  # SE SONO PASSATI GLI STIMATORI E NON QUELLI VERI
        for i in range(5):
            for j in range(4):
                if (conv_rate[i][j] > 1) or (np.isinf(conv_rate[i][j])):
                    conv_rate[i][j] = 1
    else:
        conv_rate = (conv_rate[0] * env.pois_param[0] +
                     conv_rate[1] * env.pois_param[1] +
                     conv_rate[2] * env.pois_param[2]) / np.sum(env.pois_param)

    if len(tran_prob) != 3:  # SE SONO PASSATI GLI STIMATORI E NON QUELLI VERI
        for i in range(5):
            for j in range(5):
                if (tran_prob[i][j] > 1) or (np.isinf(tran_prob[i][j])):
                    tran_prob[i][j] = 1
        tr_prob = tran_prob
    else:
        tr_prob = (tran_prob[0]*env.pois_param[0] +
                   tran_prob[1]*env.pois_param[1] +
                   tran_prob[2]*env.pois_param[2]) / np.sum(env.pois_param)

    if len(alpha) == 3:
        alphas = np.array(env.dir_params, dtype=float) / np.sum(env.dir_params)
        alphas = (alphas[0] * env.pois_param[0] +
                  alphas[1] * env.pois_param[1] +
                  alphas[2] * env.pois_param[2]) / np.sum(env.pois_param)
    else:
        alphas = alpha

    if len(n_buy) == 3:
        mepp = (env.mepp[0, :] * env.pois_param[0] +
                env.mepp[1, :] * env.pois_param[1] +
                env.mepp[2, :] * env.pois_param[2]) / np.sum(env.pois_param)
    else:
        mepp = n_buy

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
                        profits[count] = profit_puller(prices=sim_prices, conv_rate_full=conv_rate,
                                                       margins_full=env.global_margin, tran_prob=tr_prob,
                                                       alphas=alphas, mepp=mepp)
                        prices[count] = cdc(sim_prices)

                        count += 1
                sys.stdout.write('\r' + print_message + str(", pulling prices: ") + f'{count * 100 / cc} %')
    sys.stdout.write('\r' + print_message + str(", pulling prices: 100%"))
    profits = np.array(profits, dtype=float)
    best = np.argmax(profits)
    return prices[best]
