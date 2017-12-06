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

    short = Variable(torch.Tensor(short)).float()/1000

    phen = mapp.forwardtimes(short)

    return phen.data.numpy()
