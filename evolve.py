import numpy as np

from tqdm import tqdm

from deap import base
from deap import creator
from deap import tools

from pandas import DataFrame

import matplotlib.pyplot as plt

import random
import math

from toolbox import toolbox as tb
from evolve_val import evolve_val
from evolve_map import evolve_map

popsize = 30
nleg = 12
ngen_stint = 20
nstint = 20

def popval2popmap(pop):
    newpop = list()
    for p in pop:
        m = p.map
        m.val = p
        p.map = None
        newpop.append(m)

    return newpop


def popmap2popval(pop):
    newpop = list()
    for p in pop:
        v = p.val
        v.map = p
        p.val = None
        newpop.append(v)

    return newpop

def evolve():

    toolbox = tb(nleg)

    pop_val = toolbox.population_val(n=popsize)
    logbook_val = tools.Logbook()
    hof_val = tools.HallOfFame(1)

    pop_map = toolbox.population_map(n=popsize)
    logbook_map = tools.Logbook()
    hof_map = tools.HallOfFame(1)

    for m,v in zip(pop_map, pop_val):
        v.map = m

    startiter = 0

    pop = pop_val
    evolve_val(pop, ngen_stint, toolbox, logbook_val, hof_val, startiter)
    startiter += ngen_stint

    for s in tqdm(range(nstint)):
        pop = popval2popmap(pop)
        evolve_map(pop, ngen_stint, toolbox, logbook_map, hof_map, startiter)
        startiter += ngen_stint

        pop = popmap2popval(pop)
        evolve_val(pop, ngen_stint, toolbox, logbook_val, hof_val, startiter)
        startiter += ngen_stint

    pop_val = pop
    pop_map = popval2popmap(pop)

    return ((logbook_val, hof_val, pop_val), (logbook_map, logbook_map, pop_map))
