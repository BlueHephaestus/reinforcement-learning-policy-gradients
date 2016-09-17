import numpy as np

def exp_decay(initial, rate, iteration):
    #Do our k*e^(r*t) exponential decay
    return initial*np.exp(rate*iteration)

