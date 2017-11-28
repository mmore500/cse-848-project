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

import sys

# # check for proper usage
# if len(sys.argv) != 2:
#     print("Usage: python3 main.py [soduku-n]")
#     sys.exit(0)
#
# n = int(sys.argv[1])

# run evolution process
((logbook_val, hof_val), (logbook_map, logbook_map)) = evolve()

# graph map fitness vs generations
gen = logbook_map.select("gen")
fit_mins = logbook_map.select("min_novelty")
fit_maxs = logbook_map.select("max_novelty")
fit_avgs = logbook_map.select("avg_novelty")

fig, ax1 = plt.subplots()
line1 = ax1.plot(gen, fit_mins, "b-", label="Minimum Fitness")
line2 = ax1.plot(gen, fit_maxs, "g-", label="Maximum Fitness")
line3 = ax1.plot(gen, fit_avgs, "r-", label="Average Fitness")

ax1.set_xlabel("Generation")
ax1.set_ylabel("Map Novelty Fitness")
# ax1.set_ylim([0,1])

lns = line1 + line2 + line3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs)

plt.show()

# graph map fitness vs generations
gen = logbook_map.select("gen")
fit_mins = logbook_map.select("min_useful")
fit_maxs = logbook_map.select("max_useful")
fit_avgs = logbook_map.select("avg_useful")

fig, ax1 = plt.subplots()
line1 = ax1.plot(gen, fit_mins, "b-", label="Minimum Fitness")
line2 = ax1.plot(gen, fit_maxs, "g-", label="Maximum Fitness")
line3 = ax1.plot(gen, fit_avgs, "r-", label="Average Fitness")

ax1.set_xlabel("Generation")
ax1.set_ylabel("Map Useful Fitness")
# ax1.set_ylim([0,1])

lns = line1 + line2 + line3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs)

plt.show()

# graph val fitness vs generations
gen = logbook_val.select("gen")
fit_mins = logbook_val.select("min")
fit_maxs = logbook_val.select("max")
fit_avgs = logbook_val.select("avg")

fig, ax1 = plt.subplots()
line1 = ax1.plot(gen, fit_mins, "b-", label="Minimum Fitness")
line2 = ax1.plot(gen, fit_maxs, "g-", label="Maximum Fitness")
line3 = ax1.plot(gen, fit_avgs, "r-", label="Average Fitness")

ax1.set_xlabel("Generation")
ax1.set_ylabel("Val Fitness")
# ax1.set_ylim([0,1])

lns = line1 + line2 + line3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs)

plt.show()
