import numpy as np

import scipy
from scipy.stats import rv_discrete

def exp_decay(initial, rate, iteration):
    #Do our k*e^(r*t) exponential decay
    return initial*np.exp(rate*iteration)

def epsilon_greedy(epsilon):
    #Return True if exploring, False if exploiting
    r = np.random.rand(1)[0]
    if r < epsilon:
        return True
    else:
        return False

def from_discrete_dist(dist, sample_num=1):
    return rv_discrete(values=(np.arange(len(dist)), dist)).rvs(size=sample_num)

