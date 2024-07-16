import os
import itertools  # generate all parameter combinations  for parameter study
from pymatreader import read_mat
import glob
import quantities as pq
from elephant.conversion import BinnedSpikeTrain
import neo
import matplotlib.pyplot as plt
import numpy as np
import warnings


def read_dr_cell_file(full_path):
    dr_cell_struct = read_mat(full_path)

    # convert to elephant format
    TS = _extract_dr_cell_struct(dr_cell_struct)

    return TS


def _extract_dr_cell_struct(dr_cell_struct):
    try:
        TS = dr_cell_struct['SPIKEZ']['TS']
    except:
        TS = dr_cell_struct['temp']['SPIKEZ']['TS']
    return TS


def convert_to_elephant(TS):
    spiketrains_elephant = []
    for i in range(TS.shape[1]):
        spiketrain = TS[:, i]
        spiketrains_elephant.append(neo.SpikeTrain(spiketrain, t_stop=600, units='s', sort=True))
    return spiketrains_elephant


def bin_spike_trains(spiketrains, bin_size):

    bst = BinnedSpikeTrain(spiketrains, bin_size=bin_size)
    #bst.to_array()
    #bst.to_bool_array()
    bst_binary = bst.binarize()

    return bst_binary


def plot_spike_trains(spiketrains):
    spiketrains = _convert_to_numpy_and_remove_nan(spiketrains)

    # Note: elephant uses different format than DrCell (rows and columns are switched)
    for i in range(spiketrains.shape[0]):
        spiketrain = spiketrains[i, :]
        plt.eventplot(spiketrain[spiketrain != 0], lineoffsets=i, linelengths=0.75, color='black')


def plot_binned_spike_trains(binned_spiketrains):

    binned_spiketrains = binned_spiketrains.to_array()

    # Plot each binary signal separately
    for i in range(binned_spiketrains.shape[0]):
        plt.step(np.arange(len(binned_spiketrains[i])), binned_spiketrains[i] + i, where='post', label=f'Spike Train {i + 1}')


def _convert_to_numpy_and_remove_nan(spiketrains):
    spiketrains = np.asarray(spiketrains)
    spiketrains_new = np.nan_to_num(spiketrains)
    return spiketrains_new
