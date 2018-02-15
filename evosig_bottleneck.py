import numpy as np

from tqdm import tqdm

from deap import base
from deap import creator
from deap import tools

import matplotlib.pyplot as plt

import random
import math

from bottle.evolve import evolve
from bottle.evaluate_val import indirectphen, evaluate_indirect
import sys

sub_novelties = list()
sub_fits = list()

for __ in range(20):

    lb, hof, pop, mapmodel, tb = evolve()

    domphen = indirectphen(hof[0], mapmodel)

    domphen_fit = float(evaluate_indirect(hof[0], mapmodel)[0])

    subjects = [tb.clone(hof[0]) for __ in range(1000)]
    for s in subjects: tb.mutate_val(s)

    sub_phens = [indirectphen(s, mapmodel) for s in subjects]

    sub_novelties += [float(np.linalg.norm(sp - domphen)) for sp in sub_phens]

    sub_fits += [float(evaluate_indirect(s, mapmodel)[0]) - domphen_fit for s in subjects]

import json

with open('data/bottleneck_nf.json', 'w') as outfile:
    json.dump(list(zip(sub_novelties, sub_fits)), outfile)
