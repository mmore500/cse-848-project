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

# statistics
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg_novelty", lambda x: np.mean([t[0] for t in x]))
stats.register("avg_useful", lambda x: np.mean([t[1] for t in x]))
stats.register("std_novelty", lambda x: np.std([t[0] for t in x]))
stats.register("std_useful", lambda x: np.std([t[1] for t in x]))
stats.register("min_novelty", lambda x: np.min([t[0] for t in x]))
stats.register("min_useful", lambda x: np.min([t[1] for t in x]))
stats.register("max_novelty", lambda x: np.max([t[0] for t in x]))
stats.register("max_useful", lambda x: np.max([t[1] for t in x]))

def evolve_map(pop, NGEN, toolbox, logbook, hof, startiter):


    CXPB, MUTPB = 0.5, 0.2

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate_map, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in tqdm(range(NGEN)):
        # Select the next generation individuals
        offspring = toolbox.select_map(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate_map(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate_map(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate_map, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        # The population is entirely replaced by offspring
        pop[:] = offspring

        # Record data
        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=g+startiter, **record)
