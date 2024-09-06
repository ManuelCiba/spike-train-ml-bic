# TODO check if thresholding is correct
from lib.network_measures import complex_network_measures
import warnings
from lib.data_handler import folder_structure
from lib.connectivity import matrix_handling
import settings
import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# GLOBAL VARS
SOURCE_DATA_FOLDER = settings.FOLDER_NAME_matrices_scaled_independent
TARGET_DATA_FOLDER = settings.FOLDER_NAME_measures_independent


def calculate_complex_network_measures(chip_path):
    # for all chips in current folder
    file_list = sorted(glob.glob(chip_path + '/*.csv'))

    for file in file_list:

        # only calculate if result does not exist yet
        target_path = file.replace(SOURCE_DATA_FOLDER, TARGET_DATA_FOLDER)
        if os.path.isfile(target_path):
            print("Already calculated: " + target_path)
            continue  # continue with next loop iteration

        print("Calculating: " + target_path)
        matrix = np.array(matrix_handling.load_correlation_matrix(file))

        # TODO: is thresholding correct in this case?
        matrix_th = threshold1(matrix)

        # Create a graph from the connectivity matrix
        #G = nx.from_numpy_matrix(matrix)  # NetworkX
        #G = Graph.Adjacency(matrix.tolist())  # iGraph

        #matrix_scaled = complex_network_measures.normalize(matrix_th)
        #matrix_scaled_no_nan = np.nan_to_num(matrix_scaled)

        # Make the matrix symmetric by averaging corresponding elements with their counterparts across the diagonal
        #asymmetric_matrix = matrix_scaled_no_nan
        #symmetric_matrix = (asymmetric_matrix + asymmetric_matrix.T) / 2

        G = complex_network_measures.graph3(matrix_th)
        graph_list = []
        graph_list.append(G)
        measure_list, header = complex_network_measures.compute_all_features(graph_list)

        # save measure list to HD
        complex_network_measures.save_result_to_hd(target_path, measure_list, header)


def threshold1(matrix):
    try:
        list_min = []
        list_max = []
        matrix1 = np.array(matrix, dtype=float)
        scaler = StandardScaler()
        matrix1 = scaler.fit_transform(matrix1)
        matrix1 = np.nan_to_num(matrix1)
        threshold, upper, lower = 0.5, 1, 0
        matrix1[matrix1 <= threshold] = lower
        matrix1[matrix1 > threshold] = upper
        matrix1 = np.triu(matrix1) + np.triu(matrix1,1).T
        return matrix1
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None




if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    # define parameter
    bin_sizes = settings.BIN_SIZES
    window_sizes = settings.WINDOW_SIZES
    window_overlaps = settings.WINDOW_OVERLAPS
    methods = settings.CONNECTIVITY_METHODS
    groups = ["bic00", "bic10"]
    chip_names = folder_structure.get_all_chip_names()

    # base_folder = os.path.join(settings.PATH_RESULTS_FOLDER, SOURCE_DATA_FOLDER)
    # chip_path_list = folder_structure.get_all_paths(base_folder, 'bic')

    chip_path_list = folder_structure.generate_paths(target_data_folder=SOURCE_DATA_FOLDER,
                                                     methods=methods,
                                                     bin_sizes=bin_sizes,
                                                     window_sizes=window_sizes,
                                                     window_overlaps=window_overlaps,
                                                     chip_names=chip_names,
                                                     groups=groups)

    for chip_path in chip_path_list:
        print("Calculating complex network measures for Chip: " + chip_path)
        calculate_complex_network_measures(chip_path)
