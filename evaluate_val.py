import math
import numpy as np

def val2phen(val, mapp):
    ind = sum(b * 2**i for i, b in enumerate(val))
    return mapp[ind]

def evaluate_val(ind):

    return evaluate_valmap(ind, ind.map),

def evaluate_valmap(val, mapp):

    phen = val2phen(val, mapp)

    return evaluate_phen(phen)

def evaluate_phen(phen):
    return -np.std(phen)
