import numpy as np


from deap import base
from deap import creator
from deap import tools


import random
import math

from evaluate_map import evaluate_map
from evaluate_val import evaluate_val
from mutate_map import mutate_map


def toolbox(nleg):
        MAP_SIZE = 2 ** nleg
        toolbox = base.Toolbox()

        # IndividualMap
        creator.create("FitnessMap", base.Fitness, weights=(100000.0, 0.000001))
        creator.create("IndividualMap", list, fitness=creator.FitnessMap, val=None, tb=toolbox)

        def int2map(val):
            return [int(c) for c in "{0:b}".format(val).zfill(nleg)]

        toolbox.register(
                "map_attributes",
                lambda: [int2map(v) for v in range(MAP_SIZE)]
            )

        toolbox.register(
                "individual_map",
                tools.initIterate,
                creator.IndividualMap,
                toolbox.map_attributes
            )

        toolbox.register("population_map", tools.initRepeat, list, toolbox.individual_map)

        toolbox.register("mate_map", tools.cxTwoPoint)
        toolbox.register("mutate_map", mutate_map)
        toolbox.register("select_map", tools.selTournament, tournsize=5)
        toolbox.register("evaluate_map", evaluate_map)

        # IndividualVal
        creator.create("FitnessVal", base.Fitness, weights=(1.0,))
        creator.create("IndividualVal", list, fitness=creator.FitnessVal, map=None, tb=toolbox)

        toolbox.register(
                "val_attribute",
                lambda: np.random.choice([1,0])
            )

        toolbox.register(
                "individual_val",
                tools.initRepeat,
                creator.IndividualVal,
                toolbox.val_attribute,
                n=nleg
            )


        toolbox.register("population_val", tools.initRepeat, list, toolbox.individual_val)



        toolbox.register("mate_val", tools.cxTwoPoint)
        toolbox.register("mutate_val",
                tools.mutUniformInt,
                low=0,
                up=1,
                indpb=0.2
            )
        toolbox.register("select_val", tools.selTournament, tournsize=5)
        toolbox.register("evaluate_val", evaluate_val)

        return toolbox
