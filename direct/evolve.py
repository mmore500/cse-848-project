import numpy as np

from tqdm import tqdm

from deap import base
from deap import creator
from deap import tools

import matplotlib.pyplot as plt

import random
import math

from direct.toolbox import toolbox as tb

popsize = 300
nleg = 100
NGEN = 50


# statistics
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)


def evolve():

    toolbox = tb(nleg)

    pop = toolbox.population_val(n=popsize)
    logbook_val = tools.Logbook()
    hof_val = tools.HallOfFame(1)


    startiter = 0

    CXPB, MUTPB = 0.5, 0.2
    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate_val, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in tqdm(range(NGEN)):
        # Select the next generation individuals
        offspring = toolbox.select_val(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate_val(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate_val(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate_val, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by offspring
        pop[:] = offspring

        # Record data
        hof_val.update(pop)
        record = stats.compile(pop)
        logbook_val.record(gen=g+startiter, **record)

    return logbook_val, hof_val, pop, toolbox
