from deap import tools

import random
import math

import numpy as np

MUTPB = 0.2



def mutate_map(ind):
    nleg = len(ind[0])

    nmuts = np.random.binomial(len(ind), MUTPB)

    for i in np.random.choice(range(len(ind)), size=nmuts):
        tools.mutUniformInt(
                ind[i],
                low=0,
                up=1,
                indpb=0.2
            )
