import numpy as np


from deap import base
from deap import creator
from deap import tools


import random
import math

from evaluate_val import evaluate_indirect


def toolbox_indirect(maplen, mapmodel):

    toolbox = base.Toolbox()

    # IndividualVal
    creator.create("FitnessVal", base.Fitness, weights=(1.0,))
    creator.create("IndividualVal", list, fitness=creator.FitnessVal)


    toolbox.register(
            "individual_val",
            tools.initRepeat,
            creator.IndividualVal,
            np.random.uniform,
            n=maplen
        )


    toolbox.register("population_val", tools.initRepeat, list, toolbox.individual_val)


    toolbox.register("mate_val", tools.cxTwoPoint)
    toolbox.register("mutate_val", tools.mutGaussian, mu=0, sigma=0.1, indpb=1)
    toolbox.register("select_val", tools.selTournament, tournsize=5)
    toolbox.register("evaluate_val", evaluate_indirect, mapp=mapmodel)

    return toolbox
