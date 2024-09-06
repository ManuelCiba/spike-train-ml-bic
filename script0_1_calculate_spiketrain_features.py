import os
import itertools  # generate all parameter combinations  for parameter study
import glob
import quantities as pq
import warnings
from elephant.spike_train_synchrony import spike_contrast
import viziphant  # visualization for elephant results
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from lib.connectivity import correlation_methods
from lib.connectivity import matrix_handling
from lib.data_handler import spiketrain_handler
from lib.data_handler import folder_structure
from lib.data_handler import hd
import settings
import script0_2_calculate_correlation_matrices_raw as script0


TARGET_DATA_FOLDER = settings.FOLDER_NAME_synchrony


def calculate_spiketrain_features_and_save_results(method, st_list, window_size, window_overlap, folder_path):
    recording_size = st_list[0].t_stop  # use first list element to get recording duration
    step = window_size * (1 - window_overlap)
    window_start_list = list(range(0, int(recording_size), int(step)))
    window_end_list = list(range(int(window_size), int(recording_size) + int(step), int(step)))

    num_windows = min([len(window_start_list), len(window_end_list)])
    for w in range(num_windows):

        # define file name
        full_path = script0.define_full_file_name(folder_path, w)

        # only calculate if result does not exist yet
        if not os.path.isfile(full_path):
            print("Calculating spiketrain features... " + full_path)

            # get spike train for the current window
            st_w_list = [st.time_slice(t_start=window_start_list[w] * pq.s, t_stop=window_end_list[w] * pq.s) for st in st_list]

            # calculate features
            _calculate_and_plot_and_save_features(method, st_w_list, full_path)


        else:
            print("Already calculated: " + full_path)


def _calculate_and_plot_and_save_features(method, st_w_list, full_path):
    # calculate feature (result will be saved as csv, trace is used for plotting)
    df_result, fig = _calculate_feature(method, st_w_list)

    # save features
    hd.save_df_as_csv(df_result, full_path)
    hd.save_figure(fig, full_path.replace("csv", "jpg"))


def _calculate_feature(method, st_list):

    # init results, init figure
    df_result = pd.DataFrame()
    plt.figure(figsize=(8, 6))
    fig = plt.gcf()

    if method == "Spike-contrast":
        s, trace = spike_contrast(st_list, return_trace=True)  # return_trace=True: returns not only synchrony value but also curves
        result = trace.synchrony  # save whole synchrony curve, not only max value
        names = ["Spike-contrast (" + str(round(bin_size, 3)) + ")" for bin_size in trace.bin_size]
        df_result = pd.DataFrame(data=[result], columns=names)
        if not np.isnan(s):
            fig = plot_spike_contrast(trace, st_list)


    return df_result, fig


def plot_spike_contrast(trace, st_list):
    # Create a heatmap using seaborn
    plt.figure(figsize=(8, 6))
    viziphant.spike_train_synchrony.plot_spike_contrast(trace, spiketrains=st_list, c='gray', s=1)
    # Get the figure handle
    fig = plt.gcf()
    return fig







if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    # define parameter
    bin_sizes = settings.BIN_SIZES
    window_sizes = settings.WINDOW_SIZES
    window_overlaps = settings.WINDOW_OVERLAPS
    methods = ["Spike-contrast"]
    groups = ["bic00", "bic10"]

    # get all chip names
    chip_names = script0.get_all_chip_names()

    # Generate all combinations of parameters
    parameter_combinations = list(itertools.product(methods, bin_sizes, window_sizes, window_overlaps, chip_names, groups))

    # Base folder for storing results
    base_folder = os.path.join(settings.PATH_RESULTS_FOLDER, TARGET_DATA_FOLDER)

    for combination in parameter_combinations:
        method, bin_size, window_size, window_overlap, chip_name, group = combination

        # Create a folder structure based on the parameter combination
        result_folder = os.path.join(base_folder, f"method_{method}", f"bin_{bin_size}",
                                     f"window_{window_size}", f"overlap_{window_overlap}", chip_name, group)


        # 1) load file
        print("Calculating: " + chip_name + " " + group)
        spiketrains = script0.load_and_convert_file(chip_name, group)

        # 3) Calculate spike train features
        calculate_spiketrain_features_and_save_results(method, spiketrains, window_size, window_overlap, result_folder)


