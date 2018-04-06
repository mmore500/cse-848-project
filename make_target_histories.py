import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.colors as colors

import numpy as np

import json


def plot_history(seqs, title):
    fig = plt.figure()
    fig.suptitle(title, fontsize=14)
    # example data

    x = range(len(seqs[0]))
    y = [np.mean(tup) for tup in zip(*seqs)]

    # example variable error bar values
    yerr = [np.std(tup) if i % 1000 == 0 else 0 for (i, tup) in enumerate(zip(*seqs))]

    # First illustrate basic pyplot interface, using defaults where possible.
    plt.errorbar(x, y, yerr=yerr)

    plt.xlabel("Generation")
    plt.ylabel("Mean of Population Mean Leg Height")

    plt.show()

def makedf(dat, encoding):
    return pd.DataFrame.from_records([{
            'Generation' : g,
            'Table Height' : val,
            'Encoding' : encoding,
            'rep' : i
            } for i, seq in enumerate(dat) for g, val in enumerate(seq)])

def plot_history_combined(direct_seqs, bottleneck_seqs, denoising_seqs, title):

    sns.set()

    ax = sns.tsplot(data=makedf(denoising_seqs, "Denoiser"), time='Generation', unit='rep', value='Table Height', condition='Encoding', ci=95, linestyle='-', color=sns.color_palette()[0])

    ax = sns.tsplot(data=makedf(bottleneck_seqs, "Bottleneck"), time='Generation', unit='rep', value='Table Height', condition='Encoding', ci=95, linestyle=':', color=sns.color_palette()[2])

    ax = sns.tsplot(data=makedf(direct_seqs, "Direct"), time='Generation', unit='rep', value='Table Height', condition='Encoding', ci=95, linestyle='--', color=sns.color_palette()[1])

    plt.title(title)

    plt.show()

def extract(lbs):
    return [[entry['avg'] for entry in lb] for lb in lbs]

directdat = extract(json.load(open('data/direct_target_lbs.json')))
bottleneckdat = extract(json.load(open('data/bottleneck_target_lbs.json')))
noisedat = extract(json.load(open('data/noise_target_lbs.json')))

plot_history(directdat, "Direct Encoding Target History")

plot_history(bottleneckdat, "Bottleneck Encoding Target History")

plot_history(noisedat, "Denoising Encoding Target History")

plot_history_combined(directdat, bottleneckdat, noisedat, "Response to Selection for Zero Mean Leg Length")
