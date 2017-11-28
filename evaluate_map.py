import math
import numpy as np

from evaluate_val import evaluate_valmap, val2phen, evaluate_phen

nmut_vals = 100

def phendist(phen1, phen2):
    return sum(p1 == p2 for p1, p2 in zip(phen1, phen2))


def evaluate_map(ind):

    mapp = ind
    val = ind.val
    phen = val2phen(val, mapp)
    tb = ind.tb

    mut_vals = [tb.clone(val) for __ in range(nmut_vals)]
    for mv in mut_vals: tb.mutate_val(mv)

    mut_phens = [val2phen(v, mapp) for v in mut_vals]

    mut_phendists = [phendist(phen, p) for p in mut_phens]

    mut_fits = [evaluate_phen(p) for p in mut_phens]

    novelty = np.mean(mut_phendists)

    useful = np.mean(mut_fits)

    return novelty, useful
