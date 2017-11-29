import numpy as np

from tqdm import tqdm

from deap import base
from deap import creator
from deap import tools

from pandas import DataFrame

import matplotlib.pyplot as plt

import random
import math

from evolve_indirect import evolve_indirect
from evaluate_val import indirectphen, evaluate_indirect
import sys

# # check for proper usage
# if len(sys.argv) != 2:
#     print("Usage: python3 main.py [soduku-n]")
#     sys.exit(0)
#
# n = int(sys.argv[1])

sub_novelties = list()
sub_fits = list()

for __ in range(40):

    lb, hof, pop, mapmodel, tb = evolve_indirect()

    domphen = indirectphen(hof[0], mapmodel)

    subjects = [tb.clone(hof[0]) for __ in range(10000)]
    for s in subjects: tb.mutate_val(s)

    sub_phens = [indirectphen(s, mapmodel) for s in subjects]

    sub_novelties += [float(np.linalg.norm(sp - domphen)) for sp in sub_phens]

    sub_fits += [float(evaluate_indirect(s, mapmodel)[0]) for s in subjects]

import json

with open('../data/indirect_nf.json', 'w') as outfile:
    json.dump(list(zip(sub_novelties, sub_fits)), outfile)
