import numpy as np

from tqdm import tqdm

from deap import base
from deap import creator
from deap import tools

import matplotlib.pyplot as plt

import random
import math

from direct.evolve import evolve_target
import sys

lbs = list()

for __ in range(3):
    lb, hof, pop, tb = evolve_target(5000)
    print(lb)

    lbs.append(lb)

import json

with open('data/direct_target_lbs.json', 'w') as outfile:
    json.dump(lbs, outfile)