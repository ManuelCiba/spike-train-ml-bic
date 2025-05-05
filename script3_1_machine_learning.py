import warnings
warnings.filterwarnings("ignore")
from sklearn.exceptions import ConvergenceWarning
from joblib import Parallel, delayed
import multiprocessing
import os
import numpy as np
import re


from lib.machine_learning import workflow_new
from lib.data_handler import folder_structure
from lib.data_handler import hd
import settings



def withinloop(path_experiment, SOURCE_DATA_FOLDER, TARGET_DATA_FOLDER):
    source_path = path_experiment

    # check if already calculated
    target_path = source_path.replace(SOURCE_DATA_FOLDER, TARGET_DATA_FOLDER)
    full_path = os.path.join(target_path, 'df_results.pkl')
    if os.path.isfile(full_path):
        print("Already calculated: " + full_path)
        return
    else:
        print("Calculating: " + full_path)

    # load feature matrices
    full_path_bic00 = os.path.join(path_experiment, 'bic00.csv')
    full_path_bic10 = os.path.join(path_experiment, 'bic10.csv')

    # load the dataframes
    df_features_bic00 = hd.load_csv_as_df(full_path_bic00, index_col=False)
    df_features_bic10 = hd.load_csv_as_df(full_path_bic10, index_col=False)

    # get number of windows (number of all rows / 9, as we know that there are 9 chips)
    num_windows = df_features_bic10.shape[0] / 9

    # assign chip ids
    df_features_bic00 = _assign_chip_ids(df_features_bic00, num_windows)
    df_features_bic10 = _assign_chip_ids(df_features_bic10, num_windows)

    # perform ml
    df_train_results, df_test_results, X_train_list, y_train_list, X_test_list, y_test_list = workflow_new.call(models, df_features_bic00, df_features_bic10)
    workflow_new.save_df_results_to_hd(target_path,
                                       df_train_results,
                                       df_test_results,
                                       X_train_list,
                                       y_train_list,
                                       X_test_list,
                                       y_test_list)

    # save figs
    fig_test_path = os.path.join(target_path, "fig_test.pdf")
    fig_train_path = os.path.join(target_path, "fig_train.pdf")
    fig_test = workflow_new.plot_test_results(df_test_results)
    hd.save_figure(fig_test, fig_test_path)
    #hd.save_figure(fig_train, fig_train_path)

    print("Calculation finished: " + full_path)

def _get_path_experiment_list():
    # define parameter
    bin_sizes = settings.BIN_SIZES
    window_sizes = settings.WINDOW_SIZES
    window_overlaps = settings.WINDOW_OVERLAPS
    methods = settings.CONNECTIVITY_METHODS
    groups = ["bic00", "bic10"]
    chip_names = folder_structure.get_all_chip_names()

    path_experiment_list = folder_structure.generate_paths(target_data_folder=SOURCE_DATA_FOLDER,
                                                           methods=methods,
                                                           bin_sizes=bin_sizes,
                                                           window_sizes=window_sizes,
                                                           window_overlaps=window_overlaps,
                                                           chip_names=[],
                                                           groups=[])

    return path_experiment_list

def _assign_chip_ids(df, num_windows):
    # Determine the number of chips based on the size of the dataframe and number of windows per chip
    num_chips = len(df) // num_windows

    # Create an array of chip_ids, repeating each chip_id 'num_windows' times
    chip_ids = np.repeat(np.arange(1, num_chips + 1), num_windows)

    # If there are remaining rows that don't fit into a complete chip window set
    remainder = len(df) % num_windows
    if remainder > 0:
        chip_ids = np.append(chip_ids, np.repeat(num_chips + 1, remainder))

    # Add the chip_ids as a new column in the dataframe
    df['chip_id'] = chip_ids
    return df

def run_parallized(models, SOURCE_DATA_FOLDER, TARGET_DATA_FOLDER):

    # get all paths of the different parameter combinations
    path_experiment_list = _get_path_experiment_list()

    # define number of CPU cores
    num_cores = int(multiprocessing.cpu_count()/2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=Warning)
        Parallel(n_jobs=num_cores)(delayed(withinloop)(param, SOURCE_DATA_FOLDER, TARGET_DATA_FOLDER) for param in path_experiment_list)

def run(models, SOURCE_DATA_FOLDER, TARGET_DATA_FOLDER):
    base_folder = os.path.join(settings.PATH_RESULTS_FOLDER, SOURCE_DATA_FOLDER)
    path_experiment_list = folder_structure.get_all_paths(base_folder, 'overlap')


    for path_experiment in path_experiment_list:

        source_path = path_experiment

        # check if already calculated
        target_path = source_path.replace(SOURCE_DATA_FOLDER, TARGET_DATA_FOLDER)
        full_path = os.path.join(target_path, 'df_results.pkl')
        if os.path.isfile(full_path):
            print("Already calculated: " + full_path)
            continue
        else:
            print("Calculating: " + full_path)

        # load feature matrices
        full_path_bic00 = os.path.join(path_experiment, 'bic00.csv')
        full_path_bic10 = os.path.join(path_experiment, 'bic10.csv')

        # load the dataframes
        matrix_df_list_bic00 = hd.load_csv_as_df(full_path_bic00, index_col=False)
        matrix_df_list_bic10 = hd.load_csv_as_df(full_path_bic10, index_col=False)

        # transform dataframe to np (this will remove the column names)
        matrix_np_list_bic00 = np.array(matrix_df_list_bic00)
        matrix_np_list_bic10 = np.array(matrix_df_list_bic10)

        # perform ml
        df_results, X_test, y_test, = workflow_new.call(models, matrix_np_list_bic00, matrix_np_list_bic10)
        workflow_new.save_df_results_to_hd(target_path, df_results, X_test, y_test)



if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    for SOURCE_DATA_FOLDER in settings.FEATURE_SET_LIST:

        # define folder name to save results
        TARGET_DATA_FOLDER = SOURCE_DATA_FOLDER.replace("0_feature-set", "1_ML")

        # do the machine learning
        models = settings.ML_MODELS
        run_parallized(models, SOURCE_DATA_FOLDER, TARGET_DATA_FOLDER)