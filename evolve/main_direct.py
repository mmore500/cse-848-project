import numpy as np

from tqdm import tqdm

from deap import base
from deap import creator
from deap import tools

from pandas import DataFrame

import matplotlib.pyplot as plt

import random
import math

from evolve import evolve
from evaluate_val import evaluate_val
import sys

sub_novelties = list()
sub_fits = list()

for __ in range(40):
    lb, hof, pop, tb = evolve()

    domphen = hof[0]

    subjects = [tb.clone(hof[0]) for __ in range(10000)]
    for s in subjects: tb.mutate_val(s)

    sub_phens = subjects

    sub_novelties += [float(np.linalg.norm(np.array(sp) - np.array(domphen))) for sp in sub_phens]

    sub_fits += [float(evaluate_val(s)[0]) for s in subjects]


import json

with open('../data/direct_nf.json', 'w') as outfile:
    json.dump(list(zip(sub_novelties, sub_fits)), outfile)
