import numpy as np

from tqdm import tqdm

from deap import base
from deap import creator
from deap import tools

import matplotlib.pyplot as plt

import random
import math

from direct.evolve import evolve

import sys

import torch
import torchvision

def bundledata(listofpops):
    listofinds = [ind for pop in listofpops for ind in pop]
    bundled_testdata = torch.Tensor(listofinds)

    print(bundled_testdata.shape)

    torch.save(bundled_testdata, "../data/traindata.pt")


listofpops = [evolve()[2] for __ in tqdm(range(250))]

bundledata(listofpops)
