import os
import itertools  # generate all parameter combinations  for parameter study
import glob
import quantities as pq
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from lib.data_handler import spiketrain_handler
from lib.data_handler import folder_structure
from lib.data_handler import hd
import settings


TARGET_DATA_FOLDER = settings.FOLDER_NAME_split_data


def split_spiketrains_and_save_results(spiketrains, bin_size, window_size, window_overlap, result_folder_path):

    # bin spike trains
    bst = spiketrain_handler.bin_spike_trains(spiketrains, bin_size)

    # split binned spike trains into windows
    recording_size = bst.t_stop
    step = window_size * (1 - window_overlap)
    window_start_list = list(range(0, int(recording_size), int(step)))
    window_end_list = list(range(int(window_size), int(recording_size) + int(step), int(step)))

    num_windows = min([len(window_start_list), len(window_end_list)])
    for w in range(num_windows):

        # define file name
        full_path = define_full_file_name(result_folder_path, w)

        # only calculate if result does not exist yet
        if not os.path.isfile(full_path):
            print("Splitting spike trains... " + full_path)

            # get binned spike train for the current window
            bst_w = bst.time_slice(t_start=window_start_list[w] * pq.s, t_stop=window_end_list[w] * pq.s)

            # save split spike trains as an HDF5 file
            hd.save_spiketrains_as_csv(bst_w, full_path)

            # plot spike trains and save
            plot_spiketrains_and_save(bst_w, full_path)
        else:
            print("Already processed: " + full_path)


def plot_spiketrains_and_save(bst, full_path):

    fig, ax = plt.subplots()  # Create a figure and axis object

    # Heatmap for binary spike train matrix
    binary_data = bst.to_bool_array()
    sns.heatmap(binary_data, cmap='binary', cbar=False, xticklabels=False)
    ax.set_xticks([0, binary_data.shape[1] - 1])  # Start at 0, end at the number of columns (time bins)
    ax.set_xticklabels([str(bst.t_start), str(bst.t_stop)])  # Use t_start and t_stop as labels
    plt.xlabel('Time bins')
    plt.ylabel('Electrode index')
    plt.title('Binned Spike Trains')
    hd.save_figure(fig, full_path.replace("csv", "jpg"))
    fig.clear()


def get_all_chip_names():
    cwd = os.getcwd()
    path_data = cwd + '/data/MC-Japan/'

    path_chips = [x[0] for x in os.walk(path_data)]
    # remove first entry as it is the current folder
    path_chips = path_chips[1:]
    # sort by name
    path_chips = sorted(path_chips)

    chip_names = []
    for path_chip in path_chips:
        files = glob.glob(path_chip + '/*.mat')
        # Important! Sort file names, as python make wrong order
        files = sorted(files)

        # Split the path into its components
        parts = os.path.normpath(path_chip).split(os.path.sep)
        chip_name = parts[-1]

        chip_names.append(chip_name)

    return chip_names


def load_and_convert_file(chip_name, group):

    # define full path of selected chip
    path_chip = os.path.join(os.getcwd(), "data", "MC-Japan", chip_name)

    # get file name of all .mat file within folder
    files = glob.glob(path_chip + '/*.mat')
    # Important! Sort file names, as python make wrong order
    files = sorted(files)

    # first file belongs to group "no bicuculline"
    # second file belongs to group "10 mM bicuculline"
    if group == "bic00":
        TS = spiketrain_handler.read_dr_cell_file(files[0])
    if group == "bic10":
        TS = spiketrain_handler.read_dr_cell_file(files[1])

    # convert dr_cell TS file to elephant spike train file
    spike_trains_elephant = spiketrain_handler.convert_to_elephant(TS)

    return spike_trains_elephant


def define_full_file_name(result_folder, w):
    file_name = 'window' + '{:02}'.format(w)
    full_path = os.path.join(result_folder, file_name + ".csv")
    return full_path




if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    # define parameter
    bin_sizes = settings.BIN_SIZES
    window_sizes = settings.WINDOW_SIZES
    window_overlaps = settings.WINDOW_OVERLAPS
    groups = ["bic00", "bic10"]

    # get all chip names
    chip_names = get_all_chip_names()

    # Generate all combinations of parameters
    parameter_combinations = list(itertools.product(bin_sizes, window_sizes, window_overlaps, chip_names, groups))

    # Base folder for storing results
    base_folder = os.path.join(settings.PATH_RESULTS_FOLDER, TARGET_DATA_FOLDER)

    for combination in parameter_combinations:
        bin_size, window_size, window_overlap, chip_name, group = combination

        # Create a folder structure based on the parameter combination
        result_folder = os.path.join(base_folder, f"bin_{bin_size}",
                                     f"window_{window_size}", f"overlap_{window_overlap}", chip_name, group)


        # 1) load file
        print("Processing: " + chip_name + " " + group)
        spiketrains = load_and_convert_file(chip_name, group)

        # 2) split spike trains
        split_spiketrains_and_save_results(spiketrains, bin_size, window_size, window_overlap, result_folder)


