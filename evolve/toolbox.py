import numpy as np


from deap import base
from deap import creator
from deap import tools


import random
import math

from evaluate_val import evaluate_val


def toolbox(nleg):

    def initGenerator():
        cur = np.random.uniform(low=0.0, high=1000.0)
        for __ in range(nleg):
            cur += np.random.normal(loc=0, scale=1.0)
            yield cur


    toolbox = base.Toolbox()

    # IndividualVal
    creator.create("FitnessVal", base.Fitness, weights=(1.0,))
    creator.create("IndividualVal", list, fitness=creator.FitnessVal, map=None, tb=toolbox)


    toolbox.register(
            "individual_val",
            tools.initIterate,
            creator.IndividualVal,
            initGenerator
        )


    toolbox.register("population_val", tools.initRepeat, list, toolbox.individual_val)


    toolbox.register("mate_val", tools.cxTwoPoint)
    toolbox.register("mutate_val", tools.mutGaussian, mu=0, sigma=1, indpb=0.01)
    toolbox.register("select_val", tools.selTournament, tournsize=5)
    toolbox.register("evaluate_val", evaluate_val)

    return toolbox
