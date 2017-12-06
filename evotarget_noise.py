import numpy as np

from tqdm import tqdm

from deap import base
from deap import creator
from deap import tools

import matplotlib.pyplot as plt

import random
import math

from noise.evolve import evolve_target
import sys

# # check for proper usage
# if len(sys.argv) != 2:
#     print("Usage: python3 main.py [soduku-n]")
#     sys.exit(0)
#
# n = int(sys.argv[1])

lbs = list()

for __ in range(3):

    lb, hof, pop, mapmodel, tb = evolve_target(5000)

    print(lb)

    lbs.append(lb)

import json

with open('data/noise_target_nf.json', 'w') as outfile:
    json.dump(lbs, outfile)