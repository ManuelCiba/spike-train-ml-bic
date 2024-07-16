# This script reads all bicuculline data recorded from the Jimbo lab in Japan
# and calculates connectivity/correlation matrices using different methods
# (see correlation_methods.py)
# The bin sizes and bin overlap can be defined by the user


# How to create Requirements.txt
# go to terminal in pycharm
# pip install pipreqs
# pipreqs .

import os
import itertools  # generate all parameter combinations  for parameter study
import glob
import quantities as pq
import warnings

from lib.connectivity import correlation_methods
from lib.connectivity import matrix_handling
from lib.data_handler import spiketrain_handler
import settings

# GLOBAL VARS
TARGET_DATA_FOLDER = settings.FOLDER_NAME_matrices_raw


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


def calculate_correlation_and_save_results(method, bst, window_size, window_overlap, folder_path):
    recording_size = bst.t_stop
    step = window_size * (1 - window_overlap)
    window_start_list = list(range(0, int(recording_size), int(step)))
    window_end_list = list(range(int(window_size), int(recording_size) + int(step), int(step)))

    num_windows = min([len(window_start_list), len(window_end_list)])
    for w in range(num_windows):

        # define file name
        full_path = define_full_file_name(folder_path, w)

        # only calculate if result does not exist yet
        if not os.path.isfile(full_path):
            print("Calculating correlation matrix... " + full_path)

            # get binned spike train for the current window
            bst_w = bst.time_slice(t_start=window_start_list[w] * pq.s, t_stop=window_end_list[w] * pq.s)

            # calculate matrix
            matrix = _call_correlation_method(method, bst_w)

            # save csv and jpg
            matrix_handling.save_correlation_matrix(matrix, full_path)
            matrix_handling.plot_correlation_matrix(matrix, full_path.replace("csv", "jpg"))
        else:
            print("Already calculated: " + full_path)


def define_full_file_name(result_folder, w):
    file_name = 'window' + '{:02}'.format(w)
    full_path = os.path.join(result_folder, file_name + ".csv")
    return full_path


def _call_correlation_method(method, bst_w):

    matrix = 0
    if method == "tspe":
        matrix = correlation_methods.tspe_pairwise(bst_w)
    elif method == "cross_correlation":
        matrix = correlation_methods.ncc_pairwise(bst_w)
    elif method == "transfer_entropy":
        matrix = correlation_methods.te_pairwise(bst_w)
    elif method == "mutual_information":
        matrix = correlation_methods.mi_pairwise(bst_w)
    elif method == "pearson":
        matrix = correlation_methods.pearson_pairwise(bst_w)
    elif method == "spearman":
        matrix = correlation_methods.spearman_pairwise(bst_w)
    elif method == "covariance":
        matrix = correlation_methods.covariance_global(bst_w)
    elif method == "graph_lasso":
        matrix = correlation_methods.graph_lasso_global(bst_w)
    elif method == "canonical":
        matrix = correlation_methods.canonical_global(bst_w)
    elif method == "ledoit_wolf":
        matrix = correlation_methods.ledoit_wolf_global(bst_w)
    else:
        print("ERROR: Method does not exist: " + method)

    return matrix


if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    # define parameter
    bin_sizes = settings.BIN_SIZES
    window_sizes = settings.WINDOW_SIZES
    window_overlaps = settings.WINDOW_OVERLAPS
    methods = settings.CONNECTIVITY_METHODS
    groups = ["bic00", "bic10"]

    # get all chip names
    chip_names = get_all_chip_names()

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
        spiketrains = load_and_convert_file(chip_name, group)

        # 2) bin spike trains
        bst = spiketrain_handler.bin_spike_trains(spiketrains, bin_size)

        # 3) Calculate matrices
        calculate_correlation_and_save_results(method, bst, window_size, window_overlap, result_folder)





