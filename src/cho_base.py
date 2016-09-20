
import itertools#Cartesian product stuff
import numpy as np

def frange(start, end=None, inc=None):
    "A range function, that does accept float increments..."

    if end == None:
        end = start + 0.0
        start = 0.0

    if inc == None:
        inc = 1.0

    L = []
    while 1:
        next = start + len(L) * inc
        if inc > 0 and next >= end:
            break
        elif inc < 0 and next <= end:
            break
        L.append(next)
    return L

def step_roundup(num, step):
    return np.ceil(num/float(step))*float(step)

def step_rounddown(num, step):
    return np.floor(num/float(step))*float(step)

def cartesian_product(vectors):
    return [i for i in itertools.product(*vectors)]

class HyperParameter(object):
    def __init__(self, min, max, step, step_decrease_factor, stop_threshold, label):
        if min < 0:
            min = 0.0
        self.min = min
        self.max = max
        self.step = step#How much we step to get our values through the range specified by min max
        self.step_decrease_factor = step_decrease_factor#Pretty self explanatory
        self.stop_threshold = stop_threshold#When we stop decreasing
        self.label = label

    def get_vector(self):
        if self.step != 0:
            #print self.min, self.max, self.step
            #print frange(self.min, self.max+self.step, self.step)
            raw_val_vector = frange(self.min, self.max+self.step, self.step)
            val_vector = []

            #Trash our negative values, since there is currently no parameter that accepts them and still works
            for val in raw_val_vector:
                if val >= 0:
                    val_vector.append(val)

            return val_vector
        else:
            #Since if our step = 0, we only have a constant value here.
            if self.min >= 0:
                return [self.min]
            else:
                return [0.0]

