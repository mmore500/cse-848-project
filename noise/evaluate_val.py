import math
import numpy as np

import torch
from torch.autograd import Variable


def evaluate_val(ind):

    return - np.std(ind),

def evaluate_indirect(short, mapp):

    return - np.std(indirectphen(short, mapp)),

def indirectphen(short, mapp):

    short = Variable(torch.Tensor(short)).float()

    phen = mapp.forwardtimes(short)

    return phen.data.numpy()
