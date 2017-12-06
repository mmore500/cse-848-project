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

def plot_history_combined(direct_seqs, bottleneck_seqs, noise_seqs, title):

    fig = plt.figure()
    ax = plt.subplot(111)

    fig.suptitle(title, fontsize=14)

    direct_x = range(len(direct_seqs[0]))
    direct_y = [np.mean(tup) for tup in zip(*direct_seqs)]

    # example variable error bar values
    direct_yerr = [np.std(tup) if i % 1000 == 0 else 0 for (i, tup) in enumerate(zip(*direct_seqs))]

    # First illustrate basic pyplot interface, using defaults where possible.
    direct_graphic = plt.errorbar(direct_x, direct_y, yerr=direct_yerr, label="Direct")


    bottleneck_x = range(len(bottleneck_seqs[0]))
    bottleneck_y = [np.mean(tup) for tup in zip(*bottleneck_seqs)]

    # example variable error bar values
    bottleneck_yerr = [np.std(tup) if i % 1000 == 0 else 0 for (i, tup) in enumerate(zip(*bottleneck_seqs))]

    # First illustrate basic pyplot interface, using defaults where possible.
    bottleneck_graphic = plt.errorbar(bottleneck_x, bottleneck_y, yerr=bottleneck_yerr, label="Bottleneck")


    noise_x = range(len(noise_seqs[0]))
    noise_y = [np.mean(tup) for tup in zip(*noise_seqs)]

    # example variable error bar values
    noise_yerr = [np.std(tup) if i % 1000 == 0 else 0 for (i, tup) in enumerate(zip(*noise_seqs))]

    # First illustrate basic pyplot interface, using defaults where possible.
    noise_graphic = plt.errorbar(noise_x, noise_y, yerr=noise_yerr, label="Denoising")


    plt.xlabel("Generation")
    plt.ylabel("Mean of Population Mean Leg Height")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])


    plt.legend(handles=[direct_graphic, bottleneck_graphic, noise_graphic], loc='center left', bbox_to_anchor=(1, 0.5))

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
