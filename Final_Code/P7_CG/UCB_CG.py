import copy
import sys
import numpy as np
from Price_puller_CG import pull_prices, optimal_profit_lb
from Classes_CG import Hyperparameters, Day
from copy import deepcopy as cdc

class Learner:

    def __init__(self, n_arms):
        self.t = 0
        self.n_arms = n_arms
        self.tot_clicks = np.zeros(n_arms)
        self.tot_sales = np.zeros(n_arms)

    # function to reset to initial condition
    def reset(self):
        self.__init__(self.n_arms)

    # perform one step of algo to select the arm that I pull, specific of the algorithm I use
    def act(self):
        pass

    # collect output of algo and append the reward
    def update(self, arm_pulled, sales, clicks):
        self.t += 1  # update time
        self.tot_sales[arm_pulled] += sales
        self.tot_clicks[arm_pulled] += clicks


class UCB(Learner):

    def __init__(self, n_arms=4, c=1):
        super().__init__(n_arms)
        self.means = np.zeros(n_arms)  # to collect means
        self.widths = np.array([np.inf for _ in range(n_arms)])  # to collect upper confidence bounds (- mean)
        self.c = c

    # to select the arms with highest upper confidence bound
    def pull_cr(self):
        idx = np.array(self.means + self.widths, dtype=float)
        for i in range(4):
            if (idx[i] > 1) or (idx[i] == np.inf):
                idx[i] = 1
        return idx

    def update(self, arm_pulled, sales, clicks):
        super().update(arm_pulled, sales, clicks)
        # update the mean of the arm we pulled
        self.means[arm_pulled] = self.tot_sales[arm_pulled]/self.tot_clicks[arm_pulled]
        for idx in range(self.n_arms):  # for all arms, update upper confidence bounds
            n = self.tot_clicks[idx]
            if n > 0:
                self.widths[idx] = self.c*np.sqrt(2 * np.log(self.t) / n)
            else:
                self.widths[idx] = np.inf


class Items_UCB_Learner:

    def __init__(self, env, n_items=5, n_arms=4, c=1):
        self.env = env
        self.learners = [UCB(n_arms, c) for _ in range(n_items)]
        self.n_arms = n_arms
        self.n_items = n_items
        self.dirichlet = np.zeros((n_items+1))
        self.n_buys = np.zeros(n_items)
        self.count = np.zeros(n_items)
        self.total_buy = np.zeros(n_items)
        self.c = c

    def pull_prices(self, env: Hyperparameters, print_message):
        conv_rate = -1 * np.ones(shape=(5, 4))
        for i in range(5):
            conv_rate[i, :] = self.learners[i].pull_cr()
        prices = pull_prices(env=env, conv_rates=conv_rate, alpha=self.dirichlet, n_buy=self.n_buys,
                             trans_prob=env.global_transition_prob, print_message=print_message)
        return prices

    def update(self, pulled_prices, individual_clicks, individual_sales, items_sold, n_users, n_users_tot):
        for i in range(self.n_items):
            self.learners[i].update(pulled_prices[i],
                                    clicks=individual_clicks[i], sales=individual_sales[i])
            self.total_buy[i] += items_sold[i]
            self.count[i] += individual_sales[i]
            if self.count[i] > 0:
                self.n_buys[i] = (self.total_buy[i] / self.count[i]) - 1
            self.dirichlet[i + 1] += np.sum(n_users[i])
        self.dirichlet[0] += np.max([0, n_users_tot - np.sum(n_users)])

    def set_time(self, t):
        for i in range(self.n_items):
            self.learners[i].t = copy.deepcopy(t)

    def get_past_data(self, orig_individual_clicks, orig_individual_sales, orig_items_sold, orig_n_users, orig_n_users_tot, orig_time):
        individual_clicks = cdc(orig_individual_clicks)
        individual_sales = cdc(orig_individual_sales)
        items_sold = cdc(orig_items_sold)
        n_users = cdc(orig_n_users)
        n_users_tot = cdc(orig_n_users_tot)
        time = cdc(orig_time)
        for prod in range(self.n_items):
            for price in range(self.n_arms):
                self.learners[prod].tot_clicks[price] = individual_clicks[prod, price]
                self.learners[prod].tot_sales[price] = individual_sales[prod, price]
                self.learners[prod].means[price] = float(self.learners[prod].tot_sales[price] /
                                                         self.learners[prod].tot_clicks[price])
                n = self.learners[prod].tot_clicks[price]
                if n > 0:
                    self.learners[prod].widths[price] = self.c * np.sqrt(2 * np.log(time) / n)
                else:
                    self.learners[prod].widths[price] = np.inf
                self.learners[prod].t = time
            self.total_buy[prod] = items_sold[prod]
            self.count[prod] = np.sum(individual_sales[prod, :])
            if self.count[prod] > 0:
                self.n_buys[prod] = (self.total_buy[prod] / self.count[prod]) - 1
            self.dirichlet[prod + 1] += np.sum(n_users[prod])
        self.dirichlet[0] += np.max([0, n_users_tot - np.sum(n_users)])


class CG_Learner:

    def __init__(self, env, context_window=14, n_items=5, n_arms=4, c=1, ci_p=1.64):
        self.env = env
        self.n_arms = n_arms
        self.n_items = n_items
        self.t = 0
        self.ci_p = ci_p
        self.c = c
        self.context_window = context_window
        self.ass_matrix = np.zeros(shape=(2, 2), dtype=int)
        self.learners = [Items_UCB_Learner(env=self.env, n_items=n_items, n_arms=n_arms, c=self.c)]
        self.tot_click_per_type = [[np.zeros(shape=(5, 4), dtype=int) for _ in range(2)] for _ in range(2)]
        self.tot_individual_sales_per_type = [[np.zeros(shape=(5, 4), dtype=int) for _ in range(2)] for _ in range(2)]
        self.tot_items_per_type = [[np.zeros(shape=5, dtype=int) for _ in range(2)] for _ in range(2)]
        self.tot_n_users_per_type = [[np.zeros(shape=5, dtype=int) for _ in range(2)] for _ in range(2)]
        self.tot_nusers_global_per_type = [[0 for _ in range(2)] for _ in range(2)]
        self.feature_counter = np.zeros(shape=(2, 2), dtype=int)
        self.printer = ""
        self.context_history = [copy.deepcopy(self.ass_matrix)]

    def update(self, day: Day):
        self.t += 1

        for fa in range(2):
            for fb in range(2):
                self.learners[self.ass_matrix[fa, fb]].update(pulled_prices=day.pulled_prices[fa, fb, :],
                                                              individual_clicks=day.individual_clicks[fa, fb, :],
                                                              individual_sales=day.individual_sales[fa, fb, :],
                                                              items_sold=day.items_sold[fa, fb, :],
                                                              n_users=day.n_users[fa, fb, :],
                                                              n_users_tot=day.website.n_users[fa, fb])
                for prods in range(self.n_items):
                    self.tot_individual_sales_per_type[fa][fb][prods, day.pulled_prices[fa, fb, prods]] += day.individual_sales[fa, fb, prods]
                    self.tot_click_per_type[fa][fb][prods, day.pulled_prices[fa, fb, prods]] += day.individual_clicks[fa, fb, prods]
                self.feature_counter[fa, fb] += np.sum(day.n_users[fa, fb, :])
                self.tot_items_per_type[fa][fb] += day.items_sold[fa, fb, :]
                self.tot_n_users_per_type[fa][fb] += day.n_users[fa, fb, :]
                self.tot_nusers_global_per_type[fa][fb] += day.website.n_users[fa, fb]
                self.learners[self.ass_matrix[fa, fb]].set_time(t=self.t)

        if self.t % self.context_window == 0:
            self.generate_context()

    def generate_context(self):

        sys.stdout.write('\r' + self.printer + str(", generating new contexts"))
        split_a = False
        split_b = False
        split_a0_b = False
        split_a1_b = False

        p_hat_a_0 = np.sum(self.feature_counter[0, :])/np.sum(self.feature_counter)
        p_hat_a_1 = np.sum(self.feature_counter[1, :]) / np.sum(self.feature_counter)

        p_hat_a_0 = p_hat_a_0 - self.ci_p * np.sqrt(p_hat_a_0 * (1 - p_hat_a_0) / np.sum(self.feature_counter))
        p_hat_a_1 = p_hat_a_1 - self.ci_p * np.sqrt(p_hat_a_1 * (1 - p_hat_a_1) / np.sum(self.feature_counter))

        mu_hat_a_0 = self.profit_getter(a=0, print_message="Evaluating context fa=0")
        mu_hat_a_1 = self.profit_getter(a=1, print_message="Evaluating context fa=1")
        mu_hat_nosplit = self.profit_getter(print_message="Evaluating full context")

        if p_hat_a_0*mu_hat_a_0 + p_hat_a_1*mu_hat_a_1 >= mu_hat_nosplit:
            split_a = True

        if not split_a:
            p_hat_b_0 = np.sum(self.feature_counter[:, 0]) / np.sum(self.feature_counter)
            p_hat_b_1 = np.sum(self.feature_counter[:, 1]) / np.sum(self.feature_counter)

            p_hat_b_0 = p_hat_b_0 - self.ci_p * np.sqrt(p_hat_b_0 * (1 - p_hat_b_0) / np.sum(self.feature_counter))
            p_hat_b_1 = p_hat_b_1 - self.ci_p * np.sqrt(p_hat_b_1 * (1 - p_hat_b_1) / np.sum(self.feature_counter))

            mu_hat_b_0 = self.profit_getter(b=0, print_message="Evaluating context fb=0")
            mu_hat_b_1 = self.profit_getter(b=1, print_message="Evaluating context fb=1")

            # if we do not split we can use same mu_hat_nosplit as before

            if p_hat_b_0 * mu_hat_b_0 + p_hat_b_1 * mu_hat_b_1 >= mu_hat_nosplit:
                split_b = True

            if not split_b:
                self.ass_matrix = np.zeros(shape=(2, 2), dtype=int)
                self.learners = [Items_UCB_Learner(env=self.env, c=self.c)]
                clicks_mat = np.zeros(shape=(5, 4), dtype=int)
                sales_mat = np.zeros(shape=(5, 4), dtype=int)
                tot_items_buy = np.zeros(shape=5, dtype=int)
                tot_n_users = np.zeros(shape=5, dtype=int)
                tot_nusers_global = 0
                for f1 in range(2):
                    for f2 in range(2):
                        sales_mat += self.tot_individual_sales_per_type[f1][f2]
                        clicks_mat += self.tot_click_per_type[f1][f2]
                        tot_items_buy += self.tot_items_per_type[f1][f2]
                        tot_n_users += self.tot_n_users_per_type[f1][f2]
                        tot_nusers_global += self.tot_nusers_global_per_type[f1][f2]
                self.learners[0].get_past_data(clicks_mat,
                                               sales_mat,
                                               tot_items_buy,
                                               tot_n_users,
                                               tot_nusers_global, self.t)

                #for i in range(4):
                #    prices = i*np.ones(shape=5, dtype=int)
                #    self.learners[0].update(pulled_prices=prices,
                #                            individual_clicks=clicks_mat[:, i],
                #                            individual_sales=sales_mat[:, i],
                #                            items_sold=tot_items_buy_mat*(int(i == 0)),
                #                            n_users=tot_n_users*(int(i == 0)),
                #                            n_users_tot=tot_nusers_global*(int(i == 0)))
            else:
                self.ass_matrix = np.zeros(shape=(2, 2), dtype=int)
                self.ass_matrix[:, 1] = np.ones(2, dtype=int)
                self.learners = [Items_UCB_Learner(env=self.env, c=self.c), Items_UCB_Learner(env=self.env, c=self.c)]
                for lear in range(2):
                    clicks_mat = np.zeros(shape=(5, 4), dtype=int)
                    sales_mat = np.zeros(shape=(5, 4), dtype=int)
                    tot_items_buy = np.zeros(shape=5, dtype=int)
                    tot_n_users = np.zeros(shape=5, dtype=int)
                    tot_nusers_global = 0
                    for f1 in range(2):
                        sales_mat += self.tot_individual_sales_per_type[f1][lear]
                        clicks_mat += self.tot_click_per_type[f1][lear]
                        tot_items_buy += self.tot_items_per_type[f1][lear]
                        tot_n_users += self.tot_n_users_per_type[f1][lear]
                        tot_nusers_global += self.tot_nusers_global_per_type[f1][lear]
                    #for i in range(4):
                    #    prices = i * np.ones(5, dtype=int)
                    #    self.learners[lear].update(pulled_prices=prices,
                    #                               individual_clicks=clicks_mat[:, i],
                    #                               individual_sales=sales_mat[:, i],
                    #                               items_sold=tot_items_buy_mat*(int(i == 0)),
                    #                               n_users=tot_n_users*(int(i == 0)),
                    #                               n_users_tot=tot_nusers_global*(int(i == 0)))
                    self.learners[lear].get_past_data(clicks_mat, sales_mat, tot_items_buy, tot_n_users, tot_nusers_global, self.t)

        # if we split fa
        else:
            p_hat_a0_b0 = np.sum(self.feature_counter[0, 0]) / np.sum(self.feature_counter[0, :])
            p_hat_a0_b1 = np.sum(self.feature_counter[0, 1]) / np.sum(self.feature_counter[0, :])

            p_hat_a0_b0 = p_hat_a0_b0 - self.ci_p * np.sqrt(p_hat_a0_b0 * (1 - p_hat_a0_b0) /
                                                            np.sum(self.feature_counter[0, :]))
            p_hat_a0_b1 = p_hat_a0_b1 - self.ci_p * np.sqrt(p_hat_a0_b1 * (1 - p_hat_a0_b1) /
                                                            np.sum(self.feature_counter[0, :]))

            mu_hat_a0_b0 = self.profit_getter(a=0, b=0, print_message="Evaluating context fa=0, fb=0")
            mu_hat_a0_b1 = self.profit_getter(a=0, b=1, print_message="Evaluating context fa=0, fb=1")

            # if we do not split we can use mu_hat_a_0

            if p_hat_a0_b0 * mu_hat_a0_b0 + p_hat_a0_b1 * mu_hat_a0_b1 >= mu_hat_a_0:
                split_a0_b = True

            p_hat_a1_b0 = np.sum(self.feature_counter[1, 0]) / np.sum(self.feature_counter[1, :])
            p_hat_a1_b1 = np.sum(self.feature_counter[1, 1]) / np.sum(self.feature_counter[1, :])

            p_hat_a1_b0 = p_hat_a1_b0 - self.ci_p * np.sqrt(p_hat_a1_b0 * (1 - p_hat_a1_b0) /
                                                            np.sum(self.feature_counter[1, :]))
            p_hat_a1_b1 = p_hat_a1_b1 - self.ci_p * np.sqrt(p_hat_a1_b1 * (1 - p_hat_a1_b1) /
                                                            np.sum(self.feature_counter[1, :]))

            mu_hat_a1_b0 = self.profit_getter(a=1, b=0, print_message="Evaluating context fa=1, fb=0")
            mu_hat_a1_b1 = self.profit_getter(a=1, b=1, print_message="Evaluating context fa=1, fb=1")

            # if we do not split we can use mu_hat_a_1

            if p_hat_a1_b0 * mu_hat_a1_b0 + p_hat_a1_b1 * mu_hat_a1_b1 >= mu_hat_a_1:
                split_a1_b = True

            if not split_a0_b and not split_a1_b:
                self.ass_matrix = np.zeros(shape=(2, 2), dtype=int)
                self.ass_matrix[1, :] = np.ones(2, dtype=int)
                self.learners = [Items_UCB_Learner(env=self.env, c=self.c), Items_UCB_Learner(env=self.env, c=self.c)]
                for lear in range(2):
                    clicks_mat = np.zeros(shape=(5, 4), dtype=int)
                    sales_mat = np.zeros(shape=(5, 4), dtype=int)
                    tot_items_buy = np.zeros(shape=5, dtype=int)
                    tot_n_users = np.zeros(shape=5, dtype=int)
                    tot_nusers_global = 0
                    for f2 in range(2):
                        sales_mat += self.tot_individual_sales_per_type[lear][f2]
                        clicks_mat += self.tot_click_per_type[lear][f2]
                        tot_items_buy += self.tot_items_per_type[lear][f2]
                        tot_n_users += self.tot_n_users_per_type[lear][f2]
                        tot_nusers_global += self.tot_nusers_global_per_type[lear][f2]
                    #for i in range(4):
                    #    prices = i * np.ones(5, dtype=int)
                    #    self.learners[lear].update(pulled_prices=prices,
                    #                               individual_clicks=clicks_mat[:, i],
                    #                               individual_sales=sales_mat[:, i],
                    #                               items_sold=tot_items_buy_mat*(int(i == 0)),
                    #                               n_users=tot_n_users*(int(i == 0)),
                    #                               n_users_tot=tot_nusers_global*(int(i == 0)))
                    self.learners[lear].get_past_data(clicks_mat,
                                                      sales_mat,
                                                      tot_items_buy,
                                                      tot_n_users,
                                                      tot_nusers_global, self.t)

            elif split_a0_b and not split_a1_b:
                self.ass_matrix = np.zeros(shape=(2, 2), dtype=int)
                self.ass_matrix[0, 1] = 1
                self.ass_matrix[1, :] = 2*np.ones(2, dtype=int)
                self.learners = [Items_UCB_Learner(env=self.env, c=self.c),
                                 Items_UCB_Learner(env=self.env, c=self.c),
                                 Items_UCB_Learner(env=self.env, c=self.c)]
                #for i in range(4):
                #    prices = i * np.ones(5, dtype=int)
                #    self.learners[0].update(pulled_prices=prices,
                #                            individual_clicks=self.tot_click_per_type[0][0][:, prices],
                #                            individual_sales=self.tot_individual_sales_per_type[0][0][:, prices],
                #                            items_sold=self.tot_items_per_type[0][0]*(int(i == 0)),
                #                            n_users=self.tot_n_users_per_type[0][0]*(int(i == 0)),
                #                            n_users_tot=self.tot_nusers_global_per_type[0][0]*(int(i == 0)))
#
                #    self.learners[1].update(pulled_prices=prices,
                #                            individual_clicks=self.tot_click_per_type[0][1][:, prices],
                #                            individual_sales=self.tot_individual_sales_per_type[0][1][:, prices],
                #                            items_sold=self.tot_items_per_type[0][1]*(int(i == 0)),
                #                            n_users=self.tot_n_users_per_type[0][1]*(int(i == 0)),
                #                            n_users_tot=self.tot_nusers_global_per_type[0][1]*(int(i == 0)))
                self.learners[0].get_past_data(self.tot_click_per_type[0][0],
                                               self.tot_individual_sales_per_type[0][0],
                                               self.tot_items_per_type[0][0],
                                               self.tot_n_users_per_type[0][0],
                                               self.tot_nusers_global_per_type[0][0], self.t)
                self.learners[1].get_past_data(self.tot_click_per_type[0][1],
                                               self.tot_individual_sales_per_type[0][1],
                                               self.tot_items_per_type[0][1],
                                               self.tot_n_users_per_type[0][1],
                                               self.tot_nusers_global_per_type[0][1], self.t)
                clicks_mat = np.zeros(shape=(5, 4), dtype=int)
                sales_mat = np.zeros(shape=(5, 4), dtype=int)
                tot_items_buy = np.zeros(shape=5, dtype=int)
                tot_n_users = np.zeros(shape=5, dtype=int)
                tot_nusers_global = 0
                for f2 in range(2):
                    sales_mat += self.tot_individual_sales_per_type[1][f2]
                    clicks_mat += self.tot_click_per_type[1][f2]
                    tot_items_buy += self.tot_items_per_type[1][f2]
                    tot_n_users += self.tot_n_users_per_type[1][f2]
                    tot_nusers_global += self.tot_nusers_global_per_type[1][f2]
                #for i in range(4):
                #    prices = i * np.ones(5, dtype=int)
                #    self.learners[2].update(pulled_prices=prices,
                #                            individual_clicks=clicks_mat[:, i],
                #                            individual_sales=sales_mat[:, i],
                #                            items_sold=tot_items_buy_mat*(int(i == 0)),
                #                            n_users=tot_n_users*(int(i == 0)),
                #                            n_users_tot=tot_nusers_global*(int(i == 0)))
                self.learners[2].get_past_data(clicks_mat, sales_mat, tot_items_buy, tot_n_users,
                                                  tot_nusers_global, self.t)

            elif not split_a0_b and split_a1_b:
                self.ass_matrix = np.zeros(shape=(2, 2), dtype=int)
                self.ass_matrix[1, 0] = 1
                self.ass_matrix[1, 1] = 2
                self.learners = [Items_UCB_Learner(env=self.env, c=self.c),
                                 Items_UCB_Learner(env=self.env, c=self.c),
                                 Items_UCB_Learner(env=self.env, c=self.c)]
                #for i in range(4):
                #    prices = i * np.ones(5, dtype=int)
                #    self.learners[1].update(pulled_prices=prices,
                #                            individual_clicks=self.tot_click_per_type[1][0][:, prices],
                #                            individual_sales=self.tot_individual_sales_per_type[1][0][:, prices],
                #                            items_sold=self.tot_items_per_type[1][0]*(int(i == 0)),
                #                            n_users=self.tot_n_users_per_type[1][0]*(int(i == 0)),
                #                            n_users_tot=self.tot_nusers_global_per_type[1][0]*(int(i == 0)))
                #    self.learners[2].update(pulled_prices=prices,
                #                            individual_clicks=self.tot_click_per_type[1][1][:, prices],
                #                            individual_sales=self.tot_individual_sales_per_type[1][1][:, prices],
                #                            items_sold=self.tot_items_per_type[1][1]*(int(i == 0)),
                #                            n_users=self.tot_n_users_per_type[1][1]*(int(i == 0)),
                #                            n_users_tot=self.tot_nusers_global_per_type[1][1]*(int(i == 0)))
                self.learners[1].get_past_data(self.tot_click_per_type[1][0],
                                               self.tot_individual_sales_per_type[1][0],
                                               self.tot_items_per_type[1][0],
                                               self.tot_n_users_per_type[1][0],
                                               self.tot_nusers_global_per_type[1][0], self.t)
                self.learners[2].get_past_data(self.tot_click_per_type[1][1],
                                               self.tot_individual_sales_per_type[1][1],
                                               self.tot_items_per_type[1][1],
                                               self.tot_n_users_per_type[1][1],
                                               self.tot_nusers_global_per_type[1][1], self.t)
                clicks_mat = np.zeros(shape=(5, 4), dtype=int)
                sales_mat = np.zeros(shape=(5, 4), dtype=int)
                tot_items_buy = np.zeros(shape=5, dtype=int)
                tot_n_users = np.zeros(shape=5, dtype=int)
                tot_nusers_global = 0
                for f2 in range(2):
                    sales_mat += self.tot_individual_sales_per_type[0][f2]
                    clicks_mat += self.tot_click_per_type[0][f2]
                    tot_items_buy += self.tot_items_per_type[0][f2]
                    tot_n_users += self.tot_n_users_per_type[0][f2]
                    tot_nusers_global += self.tot_nusers_global_per_type[0][f2]
                #for i in range(4):
                #    prices = i * np.ones(5, dtype=int)
                #    self.learners[0].update(pulled_prices=prices,
                #                            individual_clicks=clicks_mat[:, prices],
                #                            individual_sales=sales_mat[:, prices],
                #                            items_sold=tot_items_buy_mat*(int(i == 0)),
                #                            n_users=tot_n_users*(int(i == 0)),
                #                            n_users_tot=tot_nusers_global*(int(i == 0)))
                self.learners[0].get_past_data(clicks_mat, sales_mat, tot_items_buy, tot_n_users,
                                               tot_nusers_global, self.t)

            else:
                self.ass_matrix = np.zeros(shape=(2, 2), dtype=int)
                self.ass_matrix[0, 1] = 1
                self.ass_matrix[1, 0] = 2
                self.ass_matrix[1, 1] = 3
                self.learners = [Items_UCB_Learner(env=self.env, c=self.c), Items_UCB_Learner(env=self.env, c=self.c),
                                 Items_UCB_Learner(env=self.env, c=self.c), Items_UCB_Learner(env=self.env, c=self.c)]
                #for i in range(4):
                #    prices = i * np.ones(5, dtype=int)
                #    self.learners[0].update(pulled_prices=prices,
                #                            individual_clicks=self.tot_click_per_type[0][0][:, prices],
                #                            individual_sales=self.tot_individual_sales_per_type[0][0][:, prices],
                #                            items_sold=self.tot_items_per_type[0][0]*(int(i == 0)),
                #                            n_users=self.tot_n_users_per_type[0][0]*(int(i == 0)),
                #                            n_users_tot=self.tot_nusers_global_per_type[0][0]*(int(i == 0)))
                #    self.learners[1].update(pulled_prices=prices,
                #                            individual_clicks=self.tot_click_per_type[0][1][:, prices],
                #                            individual_sales=self.tot_individual_sales_per_type[0][1][:, prices],
                #                            items_sold=self.tot_items_per_type[0][1]*(int(i == 0)),
                #                            n_users=self.tot_n_users_per_type[0][1]*(int(i == 0)),
                #                            n_users_tot=self.tot_nusers_global_per_type[0][1]*(int(i == 0)))
                #    self.learners[2].update(pulled_prices=prices,
                #                            individual_clicks=self.tot_click_per_type[1][0][:, prices],
                #                            individual_sales=self.tot_individual_sales_per_type[1][0][:, prices],
                #                            items_sold=self.tot_items_per_type[1][0]*(int(i == 0)),
                #                            n_users=self.tot_n_users_per_type[1][0]*(int(i == 0)),
                #                            n_users_tot=self.tot_nusers_global_per_type[1][0]*(int(i == 0)))
                #    self.learners[3].update(pulled_prices=prices,
                #                            individual_clicks=self.tot_click_per_type[1][1][:, prices],
                #                            individual_sales=self.tot_individual_sales_per_type[1][1][:, prices],
                #                            items_sold=self.tot_items_per_type[1][1]*(int(i == 0)),
                #                            n_users=self.tot_n_users_per_type[1][1]*(int(i == 0)),
                #                            n_users_tot=self.tot_nusers_global_per_type[1][1]*(int(i == 0)))
                self.learners[0].get_past_data(self.tot_click_per_type[0][0],
                                               self.tot_individual_sales_per_type[0][0],
                                               self.tot_items_per_type[0][0],
                                               self.tot_n_users_per_type[0][0],
                                               self.tot_nusers_global_per_type[0][0], self.t)
                self.learners[1].get_past_data(self.tot_click_per_type[0][1],
                                               self.tot_individual_sales_per_type[0][1],
                                               self.tot_items_per_type[0][1],
                                               self.tot_n_users_per_type[0][1],
                                               self.tot_nusers_global_per_type[0][1], self.t)
                self.learners[2].get_past_data(self.tot_click_per_type[1][0],
                                               self.tot_individual_sales_per_type[1][0],
                                               self.tot_items_per_type[1][0],
                                               self.tot_n_users_per_type[1][0],
                                               self.tot_nusers_global_per_type[1][0], self.t)
                self.learners[3].get_past_data(self.tot_click_per_type[1][1],
                                               self.tot_individual_sales_per_type[1][1],
                                               self.tot_items_per_type[1][1],
                                               self.tot_n_users_per_type[1][1],
                                               self.tot_nusers_global_per_type[1][1], self.t)
        #for lea in self.learners:
        #    lea.set_time(copy.deepcopy(self.t))
        self.context_history.append(copy.deepcopy(self.ass_matrix))

    def pull_prices(self, print_message):
        ret = -1*np.ones(shape=(2, 2, 5), dtype=int)
        self.printer = print_message
        prices_from_lear = []
        i = 0
        for lea in self.learners:
            i += 1
            printer = str(print_message + ", learner " + str(i))
            prices_from_lear.append(lea.pull_prices(self.env, printer))
        for f1 in range(2):
            for f2 in range(2):
                ret[f1, f2, :] = prices_from_lear[self.ass_matrix[f1, f2]]
        return ret

    def profit_getter(self, a=2, b=2, print_message="Partitioning: ") -> float:
        tot_click = np.zeros(shape=(5, 4), dtype=int)
        tot_buy = np.zeros(shape=(5, 4), dtype=int)
        meppp_den = np.zeros(shape=5, dtype=float)
        base_sales = np.zeros(shape=5, dtype=float)
        diri = np.zeros(shape=6, dtype=float)
        if a == 2:
            if b == 2:
                for f1 in range(2):
                    for f2 in range(2):
                        tot_click += cdc(self.tot_click_per_type[f1][f2])
                        tot_buy += cdc(self.tot_individual_sales_per_type[f1][f2])
                        meppp_den += cdc(np.sum(self.tot_individual_sales_per_type[f1][f2], axis=1))
                        base_sales += cdc(self.tot_items_per_type[f1][f2])
                        diri[1:6] += cdc(self.tot_n_users_per_type[f1][f2])
                        diri[0] += cdc(self.tot_nusers_global_per_type[f1][f2] - np.sum(self.tot_n_users_per_type[f1][f2]))
            else:
                for f1 in range(2):
                    tot_click += cdc(self.tot_click_per_type[f1][b])
                    tot_buy += cdc(self.tot_individual_sales_per_type[f1][b])
                    meppp_den += cdc(np.sum(self.tot_individual_sales_per_type[f1][b], axis=1))
                    base_sales += cdc(self.tot_items_per_type[f1][b])
                    diri[1:6] += cdc(self.tot_n_users_per_type[f1][b])
                    diri[0] += cdc(self.tot_nusers_global_per_type[f1][b] - np.sum(self.tot_n_users_per_type[f1][b]))
        else:
            if b == 2:
                for f2 in range(2):
                    tot_click += cdc(self.tot_click_per_type[a][f2])
                    tot_buy += cdc(self.tot_individual_sales_per_type[a][f2])
                    meppp_den += cdc(np.sum(self.tot_individual_sales_per_type[a][f2], axis=1))
                    base_sales += cdc(self.tot_items_per_type[a][f2])
                    diri[1:6] += cdc(self.tot_n_users_per_type[a][f2])
                    diri[0] += cdc(self.tot_nusers_global_per_type[a][f2] - np.sum(self.tot_n_users_per_type[a][f2]))
            else:
                tot_click += cdc(self.tot_click_per_type[a][b])
                tot_buy += cdc(self.tot_individual_sales_per_type[a][b])
                meppp_den += cdc(np.sum(self.tot_individual_sales_per_type[a][b], axis=1))
                base_sales += cdc(self.tot_items_per_type[a][b])
                diri[1:6] += cdc(self.tot_n_users_per_type[a][b])
                diri[0] += cdc(self.tot_nusers_global_per_type[a][b] - np.sum(self.tot_n_users_per_type[a][b]))

        conv_rate = np.zeros(shape=(5, 4), dtype=float)
        meppp = base_sales/meppp_den - 1
        for prod in range(5):
            for price in range(4):
                temp = (tot_buy[prod, price]/tot_click[prod, price]) - np.sqrt(np.log(self.t)/tot_click[prod, price])
                conv_rate[prod, price] = np.max([0, temp])
        prof = optimal_profit_lb(env=cdc(self.env), conv_rates=conv_rate, alpha=diri, n_buy=meppp,
                                 trans_prob=cdc(self.env.global_transition_prob), print_message=print_message)
        return prof
