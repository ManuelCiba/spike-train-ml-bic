import os
from lib.machine_learning import workflow_caroline
from lib.data_handler import folder_structure
from lib.data_handler import hd
import settings
import numpy as np
import pandas as pd

SOURCE_DATA_FOLDER = settings.FOLDER_NAME_matrices_scaled


def save_dataframe(df_bic00, df_bic10, target_path):
    full_path_bic00 = os.path.join(target_path, 'bic00.csv')
    full_path_bic10 = os.path.join(target_path, 'bic10.csv')

    hd.save_df_as_csv(df_bic00, full_path_bic00)
    hd.save_df_as_csv(df_bic10, full_path_bic10)

def merge_list_of_df_to_one_df(df_list):
    merged_df = pd.concat(df_list, axis=0, ignore_index=True)
    return merged_df

def merge_two_lists_together(list_a, list_b):
    list_final = [np.concatenate((a, b)) for a, b in zip(list_a, list_b)]
    return list_final

def merge_two_dataframes_together(df1, df2):

    merged_df = pd.concat([df1, df2], axis=1)
    return merged_df

if __name__ == '__main__':

    # define parameter
    bin_sizes = settings.BIN_SIZES
    window_sizes = settings.WINDOW_SIZES
    window_overlaps = settings.WINDOW_OVERLAPS
    methods = settings.CONNECTIVITY_METHODS
    groups = ["bic00", "bic10"]
    chip_names = folder_structure.get_all_chip_names()

    # base_folder = os.path.join(settings.PATH_RESULTS_FOLDER, SOURCE_DATA_FOLDER)
    # chip_path_list = folder_structure.get_all_paths(base_folder, 'overlap')

    path_experiment_list = folder_structure.generate_paths(target_data_folder=SOURCE_DATA_FOLDER,
                                                     methods=methods,
                                                     bin_sizes=bin_sizes,
                                                     window_sizes=window_sizes,
                                                     window_overlaps=window_overlaps,
                                                     chip_names=[],
                                                     groups=[])


    for path_experiment in path_experiment_list:

        # FEATURES: only connectivity matrices
        source_path = path_experiment
        TARGET_DATA_FOLDER = settings.FOLDER_NAME_features_matrices
        target_path = source_path.replace(SOURCE_DATA_FOLDER, TARGET_DATA_FOLDER)

        matrix_list_bic00 = workflow_caroline.load_all_matrices(source_path, 'bic00')
        matrix_list_bic10 = workflow_caroline.load_all_matrices(source_path, 'bic10')

        matrix_list_bic00 = workflow_caroline.flatten_df_matrices(matrix_list_bic00)
        matrix_list_bic10 = workflow_caroline.flatten_df_matrices(matrix_list_bic10)

        matrix_df_bic00 = merge_list_of_df_to_one_df(matrix_list_bic00)
        matrix_df_bic10 = merge_list_of_df_to_one_df(matrix_list_bic10)

        if os.path.isfile(os.path.join(target_path, 'bic10.csv')):
            print("Already calculated: " + target_path)
        else:
            save_dataframe(matrix_df_bic00, matrix_df_bic10, target_path)


        # FEATURES: only connectivity matrices (independent)
        source_path = path_experiment.replace(settings.FOLDER_NAME_matrices_scaled,
                                              settings.FOLDER_NAME_matrices_scaled_independent)
        TARGET_DATA_FOLDER = settings.FOLDER_NAME_features_matrices_independent
        target_path = source_path.replace(SOURCE_DATA_FOLDER, TARGET_DATA_FOLDER)

        matrix_list_bic00_independent = workflow_caroline.load_all_matrices(source_path, 'bic00')
        matrix_list_bic10_independent = workflow_caroline.load_all_matrices(source_path, 'bic10')

        matrix_list_bic00_independent = workflow_caroline.flatten_df_matrices(matrix_list_bic00_independent)
        matrix_list_bic10_independent = workflow_caroline.flatten_df_matrices(matrix_list_bic10_independent)

        matrix_df_bic00_independent = merge_list_of_df_to_one_df(matrix_list_bic00_independent)
        matrix_df_bic10_independent = merge_list_of_df_to_one_df(matrix_list_bic10_independent)

        if os.path.isfile(os.path.join(target_path, 'bic10.csv')):
            print("Already calculated: " + target_path)
        else:
            save_dataframe(matrix_df_bic00_independent, matrix_df_bic10_independent, target_path)


        # FEATURES: only complex network measures
        source_path = path_experiment.replace(settings.FOLDER_NAME_matrices_scaled,
                                              settings.FOLDER_NAME_measures)
        TARGET_DATA_FOLDER = settings.FOLDER_NAME_features_measures
        target_path = source_path.replace(settings.FOLDER_NAME_measures, TARGET_DATA_FOLDER)

        measures_list_bic00 = workflow_caroline.load_all_matrices(source_path, 'bic00', index_col=False)
        measures_list_bic10 = workflow_caroline.load_all_matrices(source_path, 'bic10', index_col=False)

        measures_df_bic00 = merge_list_of_df_to_one_df(measures_list_bic00)
        measures_df_bic10 = merge_list_of_df_to_one_df(measures_list_bic10)

        save_dataframe(measures_df_bic00, measures_df_bic10, target_path)


        # FEATURES: only complex network measures (independent)
        source_path = path_experiment.replace(settings.FOLDER_NAME_matrices_scaled,
                                              settings.FOLDER_NAME_measures_independent)
        TARGET_DATA_FOLDER = settings.FOLDER_NAME_features_measures_independent
        target_path = source_path.replace(settings.FOLDER_NAME_measures_independent, TARGET_DATA_FOLDER)

        measures_list_bic00 = workflow_caroline.load_all_matrices(source_path, 'bic00', index_col=False)
        measures_list_bic10 = workflow_caroline.load_all_matrices(source_path, 'bic10', index_col=False)

        measures_df_bic00_independent = merge_list_of_df_to_one_df(measures_list_bic00)
        measures_df_bic10_independent = merge_list_of_df_to_one_df(measures_list_bic10)

        save_dataframe(measures_df_bic00_independent, measures_df_bic10_independent, target_path)


        # FEATURES: complex network measures and synchrony curve
        source_path = path_experiment.replace(settings.FOLDER_NAME_matrices_scaled, settings.FOLDER_NAME_synchrony)
        # define path where features are saved (same structure, but method name is different)
        source_path_synchrony = source_path
        for connectivity_method in settings.CONNECTIVITY_METHODS:
            source_path_synchrony = source_path_synchrony.replace(connectivity_method, 'Spike-contrast')
        target_path = source_path.replace(settings.FOLDER_NAME_synchrony,
                                          settings.FOLDER_NAME_features_measures_synchrony_curve)

        #source_path_synchrony = source_path
        synchrony_curve_list_bic00 = workflow_caroline.load_all_matrices(source_path_synchrony, 'bic00', index_col=False)
        synchrony_curve_list_bic10 = workflow_caroline.load_all_matrices(source_path_synchrony, 'bic10', index_col=False)

        synchrony_curve_df_bic00 = merge_list_of_df_to_one_df(synchrony_curve_list_bic00)
        synchrony_curve_df_bic10 = merge_list_of_df_to_one_df(synchrony_curve_list_bic10)

        measures_synchrony_curve_df_bic00 = merge_two_dataframes_together(measures_df_bic00, synchrony_curve_df_bic00)
        measures_synchrony_curve_df_bic10 = merge_two_dataframes_together(measures_df_bic10, synchrony_curve_df_bic10)
        save_dataframe(measures_synchrony_curve_df_bic00, measures_synchrony_curve_df_bic10, target_path)


        # LIKE IN MARC PAPER:
        # FEATURES: complex network measures and synchrony value
        target_path = source_path.replace(settings.FOLDER_NAME_synchrony,
                                          settings.FOLDER_NAME_features_measures_synchrony_value)

        # Get S = max(s)
        synchrony_value_df_bic00 = pd.DataFrame({'Spike-Contrast (S_max)': synchrony_curve_df_bic00.max(axis=1)})
        synchrony_value_df_bic10 = pd.DataFrame({'Spike-Contrast (S_max)': synchrony_curve_df_bic10.max(axis=1)})

        measures_synchrony_value_df_bic00 = merge_two_dataframes_together(measures_df_bic00, synchrony_value_df_bic00)
        measures_synchrony_value_df_bic10 = merge_two_dataframes_together(measures_df_bic10, synchrony_value_df_bic10)
        save_dataframe(measures_synchrony_value_df_bic00, measures_synchrony_value_df_bic10, target_path)


        # FEATURES: complex network measures and synchrony value (independent)
        target_path = source_path.replace(settings.FOLDER_NAME_synchrony,
                                          settings.FOLDER_NAME_features_measures_synchrony_value_independent)

        # Get S = max(s)
        synchrony_value_df_bic00 = pd.DataFrame({'Spike-Contrast (S_max)': synchrony_curve_df_bic00.max(axis=1)})
        synchrony_value_df_bic10 = pd.DataFrame({'Spike-Contrast (S_max)': synchrony_curve_df_bic10.max(axis=1)})

        measures_synchrony_value_df_bic00_independent = merge_two_dataframes_together(measures_df_bic00_independent, synchrony_value_df_bic00)
        measures_synchrony_value_df_bic10_independent = merge_two_dataframes_together(measures_df_bic10_independent, synchrony_value_df_bic10)
        save_dataframe(measures_synchrony_value_df_bic00_independent, measures_synchrony_value_df_bic10_independent, target_path)


        # FEATURES: synchrony curves
        target_path = source_path.replace(settings.FOLDER_NAME_synchrony,
                                          settings.FOLDER_NAME_features_synchrony_curve)
        save_dataframe(synchrony_curve_df_bic00, synchrony_curve_df_bic10, target_path)

        # FEATURES: connectivity matrices and synchrony curve
        target_path = source_path.replace(settings.FOLDER_NAME_synchrony,
                                          settings.FOLDER_NAME_features_matrices_synchrony_curve)

        matrix_synchrony_curve_df_bic00 = merge_two_dataframes_together(matrix_df_bic00, synchrony_curve_df_bic00)
        matrix_synchrony_curve_df_bic10 = merge_two_dataframes_together(matrix_df_bic10, synchrony_curve_df_bic10)
        save_dataframe(matrix_synchrony_curve_df_bic00, matrix_synchrony_curve_df_bic10, target_path)

        # TODO FEATURES: all
