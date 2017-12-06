import numpy as np


from deap import base
from deap import creator
from deap import tools


import random
import math

nleg = 100

from noise.evaluate_val import evaluate_indirect, evaluate_indirect_target


def toolbox(maplen, mapmodel, target=False):


    def initGenerator():
        cur = np.random.uniform(low=0.0, high=1000.0)
        for __ in range(nleg):
            cur += np.random.normal(loc=0, scale=1.0)
            yield cur

    def initSameGenerator():
        cur = np.random.uniform(low=950.0, high=1050.0)
        for __ in range(nleg):
            yield cur + np.random.normal(loc=0, scale=10)

    toolbox = base.Toolbox()

    # IndividualVal
    creator.create("FitnessVal", base.Fitness, weights=(1.0,))
    creator.create("IndividualVal", list, fitness=creator.FitnessVal)


    if not target:
        toolbox.register(
                "individual_val",
                tools.initIterate,
                creator.IndividualVal,
                initGenerator
            )
    else:
        toolbox.register(
                "individual_val",
                tools.initIterate,
                creator.IndividualVal,
                initSameGenerator
            )


    toolbox.register("population_val", tools.initRepeat, list, toolbox.individual_val)


    toolbox.register("mate_val", tools.cxTwoPoint)
    if not target:
        toolbox.register("mutate_val", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.01)
    else:
        toolbox.register("mutate_val", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)

    toolbox.register("select_val", tools.selTournament, tournsize=5)

    if not target:
        toolbox.register("evaluate_val", evaluate_indirect, mapp=mapmodel)
    else:
        toolbox.register("evaluate_val", evaluate_indirect_target, mapp=mapmodel)

    return toolbox
