from deap import tools

import random
import math

import numpy as np

MUTPB = 0.2



def mutate_val(ind):

    nmuts = np.random.binomial(len(ind), MUTPB)
    nmuts = 1

    for i in np.random.choice(range(len(ind)), size=nmuts):
        ind[i] = int(not ind[i])
