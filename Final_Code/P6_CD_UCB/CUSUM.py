
class CUSUM:
    def __init__(self, M, eps, h):  # parameters of the method
        self.M = M
        self.eps = eps
        self.h = h
        self.t = 0
        self.reference = 0
        self.g_plus = 0
        self.g_minus = 0
    def update(self, n_cliks, n_buy):  # takes an imput sample, updates all relevant quantities
        self.t += 1
        if self.t <= self.M:
            mean = n_buy/n_cliks
            self.reference += mean / self.M
            return 0  # return zero if no change is detected
        else:
            for i in range(n_buy):
                s_plus = (1 - self.reference) - self.eps
                s_minus = -(1 - self.reference) - self.eps
                self.g_plus = max(0, self.g_plus + s_plus)
                self.g_minus = max(0, self.g_minus + s_minus)
            for j in range(n_cliks - n_buy):
                s_plus = (0 - self.reference) - self.eps
                s_minus = -(0 - self.reference) - self.eps
                self.g_plus = max(0, self.g_plus + s_plus)
                self.g_minus = max(0, self.g_minus + s_minus)
            return self.g_plus > self.h or self.g_minus > self.h  # return one when change is detected
    def reset(self):
        self.t = 0
        self.g_minus = 0
        self.g_plus = 0