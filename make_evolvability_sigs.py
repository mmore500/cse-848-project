import glob
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import json

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def plot_es(dat, title, vmax, xrang, yrang):
    fig = plt.figure()
    fig.suptitle(title, fontsize=14)
    plt.subplots_adjust(top=.92)
    hexbins = plt.hexbin(np.array([x[1] for x in dat]), np.array([x[0] for x in dat]), gridsize=20, cmap=plt.cm.Blues, norm=colors.LogNorm(vmin=None,vmax=vmax))
    bincounts = hexbins.get_array()
    plt.xlabel("Fitness Difference")
    plt.ylabel("Novelty")

    if xrang and yrang:
        plt.gca().set_xlim(xrang)
        plt.gca().set_ylim(yrang)
    plt.gca().invert_yaxis()
    plt.show()

    return bincounts

directdat = json.load(open('data/direct_nf.json'))
bottleneckdat = json.load(open('data/bottleneck_nf.json'))
noisedat = json.load(open('data/noise_nf.json'))

vmax1 = np.max(plot_es(directdat, "foobar", 100000, None, None))
vmax2 = np.max(plot_es(bottleneckdat, "foobar", 10000, None, None))
vmax3 = np.max(plot_es(noisedat, "foobar", 10000, None, None))

vmax = max(vmax1, vmax2, vmax3)

plot_es(directdat, "Direct Encoding Evolvability Signature", vmax, [-0.02, 0.015], [-0.05,0.55])

plot_es(bottleneckdat, "Bottleneck Encoding Evolvability Signature", vmax, None, None)

plot_es(noisedat, "Denoising Encoding Evolvability Signature", vmax, [-0.02, 0.015], [-0.05,0.55])

sns.set()
