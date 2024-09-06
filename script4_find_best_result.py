import os

from lib.machine_learning import workflow_caroline
from lib.data_handler import folder_structure
from lib.data_handler import hd
import settings
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # for all machine learning folder:
    for FOLDER in settings.FEATURE_SET_LIST:

        print(FOLDER)

        SOURCE_DATA_FOLDER = FOLDER.replace("0_feature-set", "1_ML")

        base_folder = os.path.join(settings.PATH_RESULTS_FOLDER, SOURCE_DATA_FOLDER)

        bin_sizes = settings.BIN_SIZES
        window_sizes = settings.WINDOW_SIZES
        window_overlaps = settings.WINDOW_OVERLAPS
        methods = settings.CONNECTIVITY_METHODS
        groups = ["bic00", "bic10"]
        chip_names = folder_structure.get_all_chip_names()

        # Create an empty list
        list_results_test = []

        for method in methods:
                for bin_size in bin_sizes:
                    for window_size in window_sizes:
                        for window_overlap in window_overlaps:

                            # define path where df_results are saved
                            path_experiment = os.path.join(base_folder,
                                                         f"method_{method}",
                                                         f"bin_{bin_size}",
                                                         f"window_{window_size}",
                                                         f"overlap_{window_overlap}")


                            # load ml results
                            df_results, X_test, y_test = workflow_caroline.load_df_results_from_hd(path_experiment)

                            # calculate test metrics
                            list_auc_test, list_accuracy_test, list_recall_test, list_precision_test = workflow_caroline.calculate_test_metrics(
                                df_results, X_test, y_test)

                            # for each ML model
                            for idx_model in range(len(df_results['Model'])):

                                # get model name
                                model = df_results['Model'][idx_model]

                                # fill in table
                                list_results_test.append({'Correlation method': method,
                                                          'Bin size': int(bin_size),
                                                          'Window size': int(window_size),
                                                          'Window overlap': window_overlap,
                                                          'ML model': model,
                                                          'AUC': list_auc_test[idx_model],
                                                          'Accuracy': list_accuracy_test[idx_model],
                                                          'Recall': list_recall_test[idx_model],
                                                          'Precision': list_precision_test[idx_model]})

        # make DataFrame and sort
        df_results_test = pd.DataFrame(list_results_test)
        df_sorted = df_results_test.sort_values(by='Accuracy', ascending=False)

        # save dataframe as csv
        full_path = os.path.join(base_folder, "results_test.csv")
        hd.save_df_as_csv(df_sorted, full_path)
