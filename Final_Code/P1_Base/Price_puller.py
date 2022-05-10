import copy

import numpy as np
import sys
from copy import deepcopy as cdc
from P1_Base.Classes_base import *

#SIMULATORE CON:
#-conversion rates
#-alpha
#-n.ro acquisti
#-transition prob.

def profit_puller(prices, MC_env: Hyperparameters, n_users_pt) -> float:

    MC_daily = Daily_Website(MC_env, cdc(prices))
    MC_daily.n_users = [n_users_pt, n_users_pt, n_users_pt]
    MC_daily.alphas = np.array(MC_env.dir_params, dtype=float)/np.sum(MC_env.dir_params)

    tran_prob = (MC_daily.transition_prob[0]+MC_daily.transition_prob[1]+MC_daily.transition_prob[2])/3
    alphas = (MC_daily.alphas[0] + MC_daily.alphas[1] + MC_daily.alphas[2]) / 3
    conv_rate = np.mean(MC_daily.conversion_rates, axis=0)

    pur_prob = np.zeros(5, dtype=float)

    temp = np.zeros(5, dtype=float)

    for p1 in range(5):
        visited = np.array([p1])

        temp[0] = conv_rate[p1]
        prob_per_p1 = np.zeros(5, dtype=float)

        for p2 in np.delete(range(5), visited):
            visited = np.array([p1, p2])

            temp[1] = conv_rate[p2]*tran_prob[p1, p2]*temp[0]
            prob_per_p1[p2] += temp[1] * (1 - prob_per_p1[p2])

            for p3 in np.delete(range(5), visited):
                visited = np.array([p1, p2, p3])

                temp[2] = conv_rate[p3]*tran_prob[p2, p3]*temp[1]
                prob_per_p1[p3] += temp[2] * (1 - prob_per_p1[p3])

                for p4 in np.delete(range(5), visited):
                    visited = np.array([p1, p2, p3, p4])

                    temp[3] = conv_rate[p4]*tran_prob[p3, p4]*temp[2]
                    prob_per_p1[p4] += temp[3] * (1 - prob_per_p1[p4])

                    for p5 in np.delete(range(5), visited):

                        temp[4] = conv_rate[p5]*tran_prob[p4, p5]*temp[3]
                        prob_per_p1[p5] += temp[4] * (1 - prob_per_p1[p5])

        prob_per_p1[p1] = conv_rate[p1]
        pur_prob += prob_per_p1*alphas[p1+1]

#    profit = 0
#    for i in range(5):
#        profit += pur_prob[i]*MC_daily.margin[i]*(1 + MC_env.mepp[i])
    profit = float(np.sum(pur_prob*MC_daily.margin*(1.0 + MC_env.mepp)))

    return profit


def pull_prices(env: Hyperparameters, conv_rates, alpha, n_buy, trans_prob, n_users_pt=100, print_message="Simulating") -> np.array:
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
        tran_prob = [tran_prob for i in range(3)]
    if len(alpha) != 3:
        alpha = [alpha for i in range(3)]

    MC_env = Hyperparameters(tran_prob, alpha, envv.pois_param, conv_rate, envv.global_margin, n_buy)

    count = 0
    cc = 4**5
    prices = [-1*np.ones(5) for i in range(cc)]
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
                        profits[count] = profit_puller(cdc(sim_prices), cdc(MC_env), n_users_pt)
                        prices[count] = cdc(sim_prices)

                        count += 1
                    sys.stdout.write('\r' + print_message + str(", pulling prices: ") + f'{(count + 1) * 100 / cc} %')

    profits = np.array(profits, dtype=float)
    best = np.argmax(profits)
    return prices[best]


#    for type_u in range(3):
#        pur_prob_per_starting_prod = []
#
#        temp = np.zeros(5, dtype=float)
#
#        for p1 in range(5):
#
#            temp[0] = copy.deepcopy(MC_daily.conversion_rates[type_u, p1])
#
#            visited = np.array([p1])
#
#            prob_per_p1 = np.zeros(5, dtype=float)
#
#            for p2 in np.delete(range(5), visited):
#                visited = np.array([p1, p2])
#
#                temp[1] = copy.deepcopy(MC_daily.conversion_rates[type_u, p2]) *\
#                          copy.deepcopy(MC_daily.transition_prob[type_u][p1, p2]) * \
#                          copy.deepcopy(temp[0])
#
#                for p3 in np.delete(range(5), visited):
#                    visited = np.array([p1, p2, p3])
#
#                    temp[2] = copy.deepcopy(MC_daily.conversion_rates[type_u, p3]) * \
#                              copy.deepcopy(MC_daily.transition_prob[type_u][p2, p3]) * \
#                              copy.deepcopy(temp[1])
#
#                    for p4 in np.delete(range(5), visited):
#                        visited = np.array([p1, p2, p3, p4])
#
#                        temp[3] = copy.deepcopy(MC_daily.conversion_rates[type_u, p4]) * \
#                                  copy.deepcopy(MC_daily.transition_prob[type_u][p3, p4]) * \
#                                  copy.deepcopy(temp[2])
#
#                        for p5 in np.delete(range(5), visited):
#
#                            temp[4] = copy.deepcopy(MC_daily.conversion_rates[type_u, p5]) * \
#                                      copy.deepcopy(MC_daily.transition_prob[type_u][p4, p5]) * \
#                                      copy.deepcopy(temp[3])
#
#                            order = [p1, p2, p3, p4, p5]
#                            for i in range(4):
#                                prob_per_p1[order[i + 1]] += temp[i + 1] * (1 - prob_per_p1[order[i + 1]])
#
#            prob_per_p1[p1] = copy.deepcopy(MC_daily.conversion_rates[type_u, p1])
#            pur_prob_per_starting_prod.append(copy.deepcopy(prob_per_p1))
#        pur_prob_type = np.zeros(5, dtype=float)
#
#        for starting_prod in range(5):
#            pur_prob_type += pur_prob_per_starting_prod[starting_prod] * \
#                             copy.deepcopy(MC_daily.alphas[type_u][starting_prod+1])
#        pur_prob_per_type.append(copy.deepcopy(pur_prob_type))
#
#    pur_prob = np.zeros(5, dtype=float)
#
#    for i in range(3):
#        pur_prob += pur_prob_per_type[i]/3


