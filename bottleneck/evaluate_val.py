import math
import numpy as np

import torch
from torch.autograd import Variable

def evaluate_indirect(short, mapp):

    return - np.std(indirectphen(short, mapp)),

def evaluate_indirect_target(short, mapp):

    p = indirectphen(short, mapp)
    return - np.std(p) - abs(np.mean(p))/10,


def indirectphen(short, mapp):

    short = Variable(torch.Tensor(short)).float()

    phen = mapp.codeforward(short) * 1000

    return phen.data.numpy()
