import copy
import numpy as np
import sys
from copy import deepcopy as cdc
from Classes_CG import Hyperparameters
from functools import reduce


# SIMULATORE CON:
# -conversion rates
# -alpha
# -n.ro acquisti
# -transition prob.


def profit_puller(prices, conv_rate_full, margins_full, tran_prob, alphas, mepp, connectivity, pois) -> float:
    conv_rate = np.zeros(shape=(3, 5), dtype=float)
    margin = np.zeros(shape=5, dtype=float)

    for j in range(5):
        for t in range(3):
            conv_rate[t, j] = conv_rate_full[t][j, prices[j]]
        margin[j] = margins_full[j, prices[j]]
    pur_prob = np.zeros(shape=(3, 5), dtype=float)
    all_prods = np.array([0, 1, 2, 3, 4])
    for p1 in range(5):
        visited = np.array([p1])
        to_visit = np.delete(copy.deepcopy(all_prods), visited)

        click_in_chain = np.zeros(shape=(3, 5), dtype=float)
        click_in_chain[0, p1] = conv_rate[0, p1]
        click_in_chain[1, p1] = conv_rate[1, p1]
        click_in_chain[2, p1] = conv_rate[2, p1]
        prob_per_p1 = np.zeros(shape= (3, 5), dtype=float)
        for p2 in np.intersect1d(connectivity[p1], to_visit):
            visited = np.array([p1, p2])
            to_visit = np.delete(copy.deepcopy(all_prods), visited)

            click_in_chain[0, p2] = conv_rate[0, p2]*tran_prob[0][p1, p2]*click_in_chain[0, p1]*(1-prob_per_p1[0, p2])
            prob_per_p1[0, p2] += click_in_chain[0, p2]
            click_in_chain[1, p2] = conv_rate[1, p2]*tran_prob[1][p1, p2]*click_in_chain[1, p1]*(1-prob_per_p1[1, p2])
            prob_per_p1[1, p2] += click_in_chain[1, p2]
            click_in_chain[2, p2] = conv_rate[2, p2]*tran_prob[2][p1, p2]*click_in_chain[2, p1]*(1-prob_per_p1[2, p2])
            prob_per_p1[2, p2] += click_in_chain[2, p2]

            for p3 in np.intersect1d(connectivity[p2], to_visit):
                visited = np.array([p1, p2, p3])
                to_visit = np.delete(copy.deepcopy(all_prods), visited)

                click_in_chain[0, p3] = conv_rate[0, p3]*tran_prob[0][p2, p3]*click_in_chain[0, p2]*(1-prob_per_p1[0, p3])
                prob_per_p1[0, p3] += click_in_chain[0, p3]
                click_in_chain[1, p3] = conv_rate[1, p3]*tran_prob[1][p2, p3]*click_in_chain[1, p2]*(1-prob_per_p1[1, p3])
                prob_per_p1[1, p3] += click_in_chain[1, p3]
                click_in_chain[2, p3] = conv_rate[2, p3]*tran_prob[2][p2, p3]*click_in_chain[2, p2]*(1-prob_per_p1[2, p3])
                prob_per_p1[2, p3] += click_in_chain[2, p3]

                for p4 in np.intersect1d(connectivity[p3], to_visit):
                    visited = np.array([p1, p2, p3, p4])
                    to_visit = np.delete(copy.deepcopy(all_prods), visited)

                    click_in_chain[0, p4] = conv_rate[0, p4]*tran_prob[0][p3, p4]*click_in_chain[0, p3]*(1-prob_per_p1[0, p4])
                    prob_per_p1[0, p4] += click_in_chain[0, p4]
                    click_in_chain[1, p4] = conv_rate[1, p4]*tran_prob[1][p3, p4]*click_in_chain[1, p3]*(1-prob_per_p1[1, p4])
                    prob_per_p1[1, p4] += click_in_chain[1, p4]
                    click_in_chain[2, p4] = conv_rate[2, p4]*tran_prob[2][p3, p4]*click_in_chain[2, p3]*(1-prob_per_p1[2, p4])
                    prob_per_p1[2, p4] += click_in_chain[2, p4]

                    for p5 in np.intersect1d(connectivity[p4], to_visit):
                        prob_per_p1[0, p5] += conv_rate[0, p5]*tran_prob[0][p4, p5]*click_in_chain[0, p4]*(1-prob_per_p1[0, p5])
                        prob_per_p1[1, p5] += conv_rate[1, p5]*tran_prob[1][p4, p5]*click_in_chain[1, p4]*(1-prob_per_p1[1, p5])
                        prob_per_p1[2, p5] += conv_rate[2, p5]*tran_prob[2][p4, p5]*click_in_chain[2, p4]*(1-prob_per_p1[2, p5])

        prob_per_p1[0, p1] = conv_rate[0, p1]
        prob_per_p1[1, p1] = conv_rate[1, p1]
        prob_per_p1[2, p1] = conv_rate[2, p1]

        pur_prob[0, :] += prob_per_p1[0, :] * (alphas[0][p1 + 1])
        pur_prob[1, :] += prob_per_p1[1, :] * (alphas[1][p1 + 1])
        pur_prob[2, :] += prob_per_p1[2, :] * (alphas[2][p1 + 1])

    profit = 0
    profit += float(np.sum(pur_prob*margin*(1.0 + mepp[0])))*pois[0]
    profit += float(np.sum(pur_prob*margin*(1.0 + mepp[1])))*pois[1]
    profit += float(np.sum(pur_prob*margin*(1.0 + mepp[2])))*pois[2]

    return profit/np.sum(pois)


def pull_prices(env, conv_rates, alpha, n_buy, trans_prob, print_message="Simulating") -> np.array:
    conv_rate = cdc(conv_rates)
    tran_prob = cdc(trans_prob)
    if len(conv_rate) != 3:  # SE SONO PASSATI GLI STIMATORI E NON QUELLI VERI
        for i in range(5):
            for j in range(4):
                if (conv_rate[i][j] > 1) or (np.isinf(conv_rate[i][j])):
                    conv_rate[i][j] = 1
        cr_rate = [conv_rate for _ in range(3)]
    else:
        cr_rate = conv_rate

    if len(tran_prob) != 3:  # SE SONO PASSATI GLI STIMATORI E NON QUELLI VERI
        for i in range(5):
            for j in range(5):
                if (tran_prob[i][j] > 1) or (np.isinf(tran_prob[i][j])):
                    tran_prob[i][j] = 1
        tr_prob = [tran_prob for _ in range(3)]
    else:
        tr_prob = tran_prob
    if len(alpha) != 3:
        alphas = [alpha/np.sum(alpha) for _ in range(3)]
    else:
        alphas = [np.zeros(6, dtype=float) for _ in range(3)]
        for j in range(3):
            alphas[j] = np.array(env.dir_params[j], dtype=float) / np.sum(env.dir_params[j])

    if len(n_buy) != 3:
        n_buys = [n_buy for _ in range(3)]
    else:
        n_buys = n_buy

    connectivity = np.zeros(shape=(5, 2), dtype=int)
    for j in range(5):
        connectivity[j, :] = reduce(np.union1d, (np.array(np.where(tran_prob[0][j, :] > 0)),
                                                 np.array(np.where(tran_prob[1][j, :] > 0)),
                                                 np.array(np.where(tran_prob[2][j, :] > 0))))

    count = 0
    cc = 4**5
    prices = -1*np.ones(shape=(cc, 5), dtype=int)
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
                        profits[count] = profit_puller(prices=sim_prices, conv_rate_full=cr_rate,
                                                       margins_full=env.global_margin, tran_prob=tr_prob,
                                                       alphas=alphas, mepp=n_buys, connectivity=connectivity,
                                                       pois=env.pois_param)
                        prices[count, :] = cdc(sim_prices)

                        count += 1
                sys.stdout.write('\r' + print_message + str(", pulling prices: ") + f'{count * 100 / cc} %')
    sys.stdout.write('\r' + print_message + str(", pulling prices: 100%"))
    profits = np.array(profits, dtype=float)
    best = np.argmax(profits)
    return prices[best]


def optimal_profit_lb(env: Hyperparameters, conv_rates, alpha, n_buy, trans_prob, print_message="Simulating")\
        -> float:
    conv_rate = cdc(conv_rates)
    tran_prob = cdc(trans_prob)
    if len(conv_rate) != 3:  # SE SONO PASSATI GLI STIMATORI E NON QUELLI VERI
        for i in range(5):
            for j in range(4):
                if (conv_rate[i][j] > 1) or (np.isinf(conv_rate[i][j])):
                    conv_rate[i][j] = 1
        cr_rate = [conv_rate for _ in range(3)]
    else:
        cr_rate = conv_rate

    if len(tran_prob) != 3:  # SE SONO PASSATI GLI STIMATORI E NON QUELLI VERI
        for i in range(5):
            for j in range(5):
                if (tran_prob[i][j] > 1) or (np.isinf(tran_prob[i][j])):
                    tran_prob[i][j] = 1
        tr_prob = [tran_prob for _ in range(3)]
    else:
        tr_prob = tran_prob
    if len(alpha) != 3:
        alphas = [alpha / np.sum(alpha) for _ in range(3)]
    else:
        alphas = [np.zeros(6, dtype=float) for _ in range(3)]
        for j in range(3):
            alphas[j] = np.array(env.dir_params[j], dtype=float) / np.sum(env.dir_params[j])

    if len(n_buy) != 3:
        n_buys = [n_buy for _ in range(3)]
    else:
        n_buys = n_buy

    connectivity = np.zeros(shape=(5, 2), dtype=int)
    for j in range(5):
        connectivity[j, :] = reduce(np.union1d, (np.array(np.where(tran_prob[0][j, :] > 0)),
                                                 np.array(np.where(tran_prob[1][j, :] > 0)),
                                                 np.array(np.where(tran_prob[2][j, :] > 0))))

    count = 0
    cc = 4 ** 5
    prices = -1 * np.ones(shape=(cc, 5), dtype=int)
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
                        profits[count] = profit_puller(prices=sim_prices, conv_rate_full=cr_rate,
                                                       margins_full=env.global_margin, tran_prob=tr_prob,
                                                       alphas=alphas, mepp=n_buys, connectivity=connectivity,
                                                       pois=env.pois_param)
                        prices[count] = cdc(sim_prices)

                        count += 1
                sys.stdout.write('\r' + print_message + str(", pulling prices: ") + f'{count * 100 / cc} %')
    sys.stdout.write('\r' + print_message + str(", pulling prices: 100%"))
    profits = np.array(profits, dtype=float)
    best = np.argmax(profits)
    return profits[best]
