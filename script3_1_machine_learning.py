import warnings
warnings.filterwarnings("ignore")
from sklearn.exceptions import ConvergenceWarning
from joblib import Parallel, delayed
import multiprocessing
import os
import numpy as np


from lib.machine_learning import workflow_caroline
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
    matrix_df_list_bic00 = hd.load_csv_as_df(full_path_bic00, index_col=False)
    matrix_df_list_bic10 = hd.load_csv_as_df(full_path_bic10, index_col=False)

    # transform dataframe to np (this will remove the column names)
    matrix_np_list_bic00 = np.array(matrix_df_list_bic00)
    matrix_np_list_bic10 = np.array(matrix_df_list_bic10)

    # perform ml
    df_results, X_test, y_test, = workflow_caroline.caroline_workflow(models, matrix_np_list_bic00,
                                                                      matrix_np_list_bic10)
    workflow_caroline.save_df_results_to_HD(target_path, df_results, X_test, y_test)


def run_parallized(models, SOURCE_DATA_FOLDER, TARGET_DATA_FOLDER):
    base_folder = os.path.join(settings.PATH_RESULTS_FOLDER, SOURCE_DATA_FOLDER)
    path_experiment_list = folder_structure.get_all_paths(base_folder, 'overlap')
    num_cores = int(multiprocessing.cpu_count()/4)

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
        df_results, X_test, y_test, = workflow_caroline.caroline_workflow(models, matrix_np_list_bic00, matrix_np_list_bic10)
        workflow_caroline.save_df_results_to_HD(target_path, df_results, X_test, y_test)



if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    for SOURCE_DATA_FOLDER in settings.FEATURE_SET_LIST:

        # define folder name to save results
        TARGET_DATA_FOLDER = SOURCE_DATA_FOLDER.replace("0_feature-set", "1_ML")

        # do the machine learning
        models = settings.ML_MODELS
        run_parallized(models, SOURCE_DATA_FOLDER, TARGET_DATA_FOLDER)