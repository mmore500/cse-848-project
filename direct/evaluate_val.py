import math
import numpy as np

import torch
from torch.autograd import Variable


def evaluate_val(ind):

    return - np.std(ind),


def evaluate_val_target(ind):

    return - np.std(ind) - abs(np.mean(ind))/10,
