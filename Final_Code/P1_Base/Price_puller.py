import copy
import numpy as np
import sys
from copy import deepcopy as cdc
from functools import reduce
from P1_Base.Classes_base import Hyperparameters, Day

# SIMULATORE CON:
# -conversion rates
# -alpha
# -n.ro acquisti
# -transition prob.

# we replaced the for loop with a full enumeration for computational efficiency, it should be better to use a for cycle
# but instead of running in 12 second per call it now works on 4.5
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
        prob_per_p1 = np.zeros(shape=(3, 5), dtype=float)

        for p2 in np.intersect1d(connectivity[p1], to_visit):
            visited = np.array([p1, p2])
            to_visit = np.delete(copy.deepcopy(all_prods), visited)

            click_in_chain[0, p2] = conv_rate[0, p2]*tran_prob[0][p1, p2]*click_in_chain[0, p1]*(1-prob_per_p1[0, p2])
            prob_per_p1[0, p2] += cdc(click_in_chain[0, p2])
            click_in_chain[1, p2] = conv_rate[1, p2]*tran_prob[1][p1, p2]*click_in_chain[1, p1]*(1-prob_per_p1[1, p2])
            prob_per_p1[1, p2] += cdc(click_in_chain[1, p2])
            click_in_chain[2, p2] = conv_rate[2, p2]*tran_prob[2][p1, p2]*click_in_chain[2, p1]*(1-prob_per_p1[2, p2])
            prob_per_p1[2, p2] += cdc(click_in_chain[2, p2])

            for p3 in np.intersect1d(connectivity[p2], to_visit):
                visited = np.array([p1, p2, p3])
                to_visit = np.delete(copy.deepcopy(all_prods), visited)

                click_in_chain[0, p3] = conv_rate[0, p3]*tran_prob[0][p2, p3]*click_in_chain[0, p2]*(1-prob_per_p1[0, p3])
                prob_per_p1[0, p3] += cdc(click_in_chain[0, p3])
                click_in_chain[1, p3] = conv_rate[1, p3]*tran_prob[1][p2, p3]*click_in_chain[1, p2]*(1-prob_per_p1[1, p3])
                prob_per_p1[1, p3] += cdc(click_in_chain[1, p3])
                click_in_chain[2, p3] = conv_rate[2, p3]*tran_prob[2][p2, p3]*click_in_chain[2, p2]*(1-prob_per_p1[2, p3])
                prob_per_p1[2, p3] += cdc(click_in_chain[2, p3])

                for p4 in np.intersect1d(connectivity[p3], to_visit):
                    visited = np.array([p1, p2, p3, p4])
                    to_visit = np.delete(copy.deepcopy(all_prods), visited)

                    click_in_chain[0, p4] = conv_rate[0, p4]*tran_prob[0][p3, p4]*click_in_chain[0, p3]*(1-prob_per_p1[0, p4])
                    prob_per_p1[0, p4] += cdc(click_in_chain[0, p4])
                    click_in_chain[1, p4] = conv_rate[1, p4]*tran_prob[1][p3, p4]*click_in_chain[1, p3]*(1-prob_per_p1[1, p4])
                    prob_per_p1[1, p4] += cdc(click_in_chain[1, p4])
                    click_in_chain[2, p4] = conv_rate[2, p4]*tran_prob[2][p3, p4]*click_in_chain[2, p3]*(1-prob_per_p1[2, p4])
                    prob_per_p1[2, p4] += cdc(click_in_chain[2, p4])

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
    profit += float(np.sum(pur_prob[0, :]*margin*(1.0 + mepp[0, :])))*float(pois[0])
    profit += float(np.sum(pur_prob[1, :]*margin*(1.0 + mepp[1, :])))*float(pois[1])
    profit += float(np.sum(pur_prob[2, :]*margin*(1.0 + mepp[2, :])))*float(pois[2])

    return float(profit/float(np.sum(pois)))


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
    profits = np.zeros(cc, dtype=float)

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
                        profits[count] = profit_puller(prices=sim_prices,
                                                       conv_rate_full=cr_rate,
                                                       margins_full=env.global_margin,
                                                       tran_prob=tr_prob,
                                                       alphas=alphas,
                                                       mepp=n_buys,
                                                       connectivity=connectivity,
                                                       pois=env.pois_param)
                        prices[count, :] = cdc(sim_prices)

                        count += 1
                sys.stdout.write('\r' + print_message + str(", pulling prices: ") + f'{count * 100 / cc} %')
    sys.stdout.write('\r' + print_message + str(", pulling prices: 100%"))
    best = np.argmax(profits)
    return prices[best]


def profit_puller_old(prices, conv_rate_full, margins_full, tran_prob, alphas, mepp, connectivity) -> float:

    conv_rate = np.zeros(shape=5, dtype=float)
    margin = np.zeros(shape=5, dtype=float)

    for j in range(5):
        conv_rate[j] = conv_rate_full[j, prices[j]]
        margin[j] = margins_full[j, prices[j]]

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


def pull_prices_old(env, conv_rates, alpha, n_buy, trans_prob, print_message="Simulating") -> np.array:
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
        alphas = copy.deepcopy(alpha)
        for j in range(3):
            alphas[j] = np.array(env.dir_params[j], dtype=float) / np.sum(env.dir_params[j])
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

    connectivity = np.zeros(shape=(5, 2), dtype=int)
    for j in range(5):
        connectivity[j, :] = np.array(np.where(tr_prob[j, :] > 0))

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
                        profits[count] = profit_puller_old(prices=sim_prices, conv_rate_full=conv_rate,
                                                       margins_full=env.global_margin, tran_prob=tr_prob,
                                                       alphas=alphas, mepp=mepp, connectivity=connectivity)
                        prices[count, :] = cdc(sim_prices)

                        count += 1
                sys.stdout.write('\r' + print_message + str(", pulling prices: ") + f'{count * 100 / cc} %')
    sys.stdout.write('\r' + print_message + str(", pulling prices: 100%"))
    profits = np.array(profits, dtype=float)
    best = np.argmax(profits)
    return prices[best, :]

def pull_prices_explor(env, conv_rates, alpha, n_buy, trans_prob, print_message="Simulating") -> np.array:
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
    profits = np.zeros(cc, dtype=float)

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
                        profits[count] = profit_puller(prices=sim_prices,
                                                       conv_rate_full=cr_rate,
                                                       margins_full=env.global_margin,
                                                       tran_prob=tr_prob,
                                                       alphas=alphas,
                                                       mepp=n_buys,
                                                       connectivity=connectivity,
                                                       pois=env.pois_param)
                        prices[count, :] = cdc(sim_prices)

                        count += 1
                sys.stdout.write('\r' + print_message + str(", pulling prices: ") + f'{count * 100 / cc} %')
    sys.stdout.write('\r' + print_message + str(", pulling prices: 100%"))
    return profits


def expected_profits(env, conv_rates, alpha, n_buy, trans_prob, print_message="Simulating") -> np.array:
    cr_rate = cdc(conv_rates)
    tr_prob = cdc(trans_prob)

    alphas = [np.zeros(6, dtype=float) for _ in range(3)]
    for j in range(3):
        alphas[j] = np.array(env.dir_params[j], dtype=float) / np.sum(env.dir_params[j])

    n_buys = n_buy

    connectivity = np.zeros(shape=(5, 2), dtype=int)
    for j in range(5):
        connectivity[j, :] = reduce(np.union1d, (np.array(np.where(tr_prob[0][j, :] > 0)),
                                                 np.array(np.where(tr_prob[1][j, :] > 0)),
                                                 np.array(np.where(tr_prob[2][j, :] > 0))))

    count = 0
    cc = 4**5
    prices = -1*np.ones(shape=(cc, 5), dtype=int)
    profits = np.zeros(shape=(4, 4, 4, 4, 4), dtype=float)

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
                        profits[p1, p2, p3, p4, p5] = profit_puller(prices=sim_prices,
                                                                    conv_rate_full=cr_rate,
                                                                    margins_full=env.global_margin,
                                                                    tran_prob=tr_prob,
                                                                    alphas=alphas,
                                                                    mepp=n_buys,
                                                                    connectivity=connectivity,
                                                                    pois=env.pois_param)
                        prices[count, :] = cdc(sim_prices)

                        count += 1
                sys.stdout.write('\r' + print_message + str(", computing expected prices: ") + f'{count * 100 / cc} %')
    sys.stdout.write('\r' + print_message + str(", computing expected prices: 100%"))
    return profits

def greedy_pull_price_exact(env, conv_rates, alpha, n_buy, trans_prob):
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

    prices = np.zeros(5, dtype=int)
    current_profit = profit_puller(prices=prices,
                                   conv_rate_full=cr_rate,
                                   margins_full=env.global_margin,
                                   tran_prob=tr_prob,
                                   alphas=alphas,
                                   mepp=n_buys,
                                   connectivity=connectivity,
                                   pois=env.pois_param)

    converged = False
    while converged == False:
        converged, prices, current_profit = greedy_step_MC(prices=prices,
                                                        conv_rate_full=cr_rate,
                                                        margins_full=env.global_margin,
                                                        tran_prob=tr_prob,
                                                        alphas=alphas,
                                                        mepp=n_buys,
                                                        connectivity=connectivity,
                                                        pois=env.pois_param,
                                                        curr_prof=current_profit)
    return prices

def greedy_step(prices, conv_rate_full, margins_full, tran_prob, alphas, mepp, connectivity, pois, curr_prof):
    greedy_profits = np.zeros(5, dtype=float)
    for i in range(5):  # for every product we raise one price at a time
        new_prices = copy.deepcopy(prices)
        if new_prices[i] != 3:
            new_prices[i] += 1
            greedy_profits[i] = profit_puller(prices=new_prices,
                                              conv_rate_full=conv_rate_full,
                                              margins_full=margins_full,
                                              tran_prob=tran_prob,
                                              alphas=alphas,
                                              mepp=mepp,
                                              connectivity=connectivity,
                                              pois=pois)
        else:
            greedy_profits[i] = curr_prof
    greedy_delta = np.zeros(5, dtype=float)
    for i in range(5):
        greedy_delta[i] = greedy_profits[i]-curr_prof
    if np.max(greedy_delta) > 0:
        best_product = np.argmax(greedy_delta)
        best_prices = copy.deepcopy(prices)
        best_prices[best_product] += 1
        prof = greedy_profits[best_product]
        check_conv = False
    else:
        check_conv = True
        best_prices = copy.deepcopy(prices)
        prof = curr_prof
    return check_conv, best_prices, prof

def greedy_step_MC(prices, conv_rate_full, margins_full, tran_prob, alphas, mepp, connectivity, pois, curr_prof):
    greedy_profits = np.zeros(5, dtype=float)
    env = Hyperparameters(tran_prob, alphas, pois, conv_rate_full, margins_full,
                          mean_extra_purchases_per_product=mepp)

    for i in range(5):  # for every product we raise one price at a time
        new_prices = copy.deepcopy(prices)
        if new_prices[i] != 3:
            new_prices[i] += 1
            day = Day(env, new_prices)
            day.run_simulation()
            greedy_profits[i] = day.profit / np.sum(day.n_users)  # normalized profit
        else:
            greedy_profits[i] = curr_prof
    greedy_delta = np.zeros(5, dtype=float)
    for i in range(5):
        greedy_delta[i] = greedy_profits[i]-curr_prof
    if np.max(greedy_delta) > 0:
        best_product = np.argmax(greedy_delta)
        best_prices = copy.deepcopy(prices)
        best_prices[best_product] += 1
        prof = greedy_profits[best_product]
        check_conv = False
    else:
        check_conv = True
        best_prices = copy.deepcopy(prices)
        prof = curr_prof
    return check_conv, best_prices, prof
