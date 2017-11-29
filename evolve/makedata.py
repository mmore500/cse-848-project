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
from bundledata import bundledata

import sys

# # check for proper usage
# if len(sys.argv) != 2:
#     print("Usage: python3 main.py [soduku-n]")
#     sys.exit(0)
#
# n = int(sys.argv[1])


listofpops = [evolve()[2] for __ in tqdm(range(250))]

bundledata(listofpops)
