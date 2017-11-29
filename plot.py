import matplotlib.pyplot as plt

import matplotlib.colors as colors

import numpy as np

import json

def plot_es(dat, title, vmax):
    fig = plt.figure()
    fig.suptitle(title, fontsize=14)
    plt.subplots_adjust(top=.92)
    hexbins = plt.hexbin(np.array([x[1] for x in dat]), -np.array([x[0] for x in dat]), gridsize=20, cmap=plt.cm.Blues, norm=colors.LogNorm(vmin=None,vmax=vmax))
    bincounts = hexbins.get_array()
    plt.xlabel("Fitness")
    plt.ylabel("Novelty")
    cb = fig.colorbar(hexbins)
    cb.set_label('$\log_{10}(N)$')
    axes = plt.gca()
    axes.set_xlim([-1,0])
    axes.set_ylim([-3000,0])
    plt.show()
    return bincounts

directdat = json.load(open('data/direct_nf.json'))
indirectdat = json.load(open('data/indirect_nf.json'))

vmax1 = np.max(plot_es(directdat, "foobar", 10000))
vmax2 = np.max(plot_es(indirectdat, "foobar", 10000))

vmax = max(vmax1, vmax2)

plot_es(directdat, "Direct Encoding Evolvability Signature", vmax)

plot_es(indirectdat, "Indirect Encoding Evolvability Signature", vmax)
