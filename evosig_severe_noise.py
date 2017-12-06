import numpy as np

from tqdm import tqdm

from deap import base
from deap import creator
from deap import tools

import matplotlib.pyplot as plt

import random
import math

from noise.evolve import evolve
from noise.evaluate_val import indirectphen, evaluate_indirect
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

    lb, hof, pop, mapmodel, tb = evolve()

    domphen = indirectphen(hof[0], mapmodel)

    subjects = [tb.clone(hof[0]) for __ in range(10000)]
    for s in subjects:
        for __ in range(100):
            tb.mutate_val(s)

    sub_phens = [indirectphen(s, mapmodel) for s in subjects]

    sub_novelties += [float(np.linalg.norm(sp - domphen)) for sp in sub_phens]

    sub_fits += [float(evaluate_indirect(s, mapmodel)[0]) for s in subjects]

import json

with open('data/noise_severe_nf.json', 'w') as outfile:
    json.dump(list(zip(sub_novelties, sub_fits)), outfile)
