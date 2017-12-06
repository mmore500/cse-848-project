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
    plt.show()

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
    axes.set_xlim([-2,0])
    axes.set_ylim([-2,0])
    plt.show()
    return bincounts

directdat = json.load(open('data/direct_nf.json'))
bottleneckdat = json.load(open('data/bottleneck_nf.json'))
noisedat = json.load(open('data/noise_nf.json'))
direct_severe_dat = json.load(open('data/direct_severe_nf.json'))
noise_severe_dat = json.load(open('data/noise_severe_nf.json'))

vmax1 = np.max(plot_es(directdat, "foobar", 10000))
vmax2 = np.max(plot_es(bottleneckdat, "foobar", 10000))
vmax3 = np.max(plot_es(noisedat, "foobar", 10000))
vmax4 = np.max(plot_es(direct_severe_dat, "foobar", 10000))
vmax5 = np.max(plot_es(noise_severe_dat, "foobar", 10000))

vmax = max(vmax1, vmax2, vmax3, vmax4, vmax5)

plot_es(directdat, "Direct Encoding Evolvability Signature", vmax)

plot_es(bottleneckdat, "Bottleneck Encoding Evolvability Signature", vmax)

plot_es(noisedat, "Denoising Encoding Evolvability Signature", vmax)

plot_es(direct_severe_dat, "Direct Encoding Severe Evolvability Signature", vmax)

plot_es(noise_severe_dat, "Denoising Encoding Severe Evolvability Signature", vmax)
