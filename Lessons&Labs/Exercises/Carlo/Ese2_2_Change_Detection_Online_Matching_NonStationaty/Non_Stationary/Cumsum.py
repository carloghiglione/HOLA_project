# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 17:19:12 2022

@author: carlo
"""
# CUMSM method to identify changepoint in a given univariate time series
class CUMSUM:
    def __init__(self, M, eps, h):  # parameters of the method
        self.M = M
        self.eps = eps
        self.h = h
        self.t = 0
        self.reference = 0
        self.g_plus = 0
        self.g_minus = 0
        
    def update(self, sample):   # takes an imput sample, updates all relevant quantities
        self.t += 1
        if self.t <= self.M:
            self.reference += sample/self.M
            return 0            # return zero if no change is detected
        else:
            s_plus = (sample - self.reference) - self.eps
            s_minus = -(sample - self.reference) - self.eps
            self.g_plus = max(0, self.g_plus + s_plus)
            self.g_minus = max(0, self.g_minus + s_minus)
            return self.g_plus > self.h or self.g_minus > self.h     # return one when change is detected
        
    def reset(self):
        self.t = 0
        self.g_minus = 0
        self.g_plus = 0