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
        path_experiment_list = folder_structure.get_all_paths(base_folder, 'overlap')

        # Create an empty DataFrame
        df_results_test = pd.DataFrame(columns=['Parameter', 'AUC', 'Accuracy', 'Recall', 'Precision'])
        list_results_test = []

        # for all experiments (=different parameter)
        for path_experiment in path_experiment_list:

            # load ml results
            df_results, X_test, y_test = workflow_caroline.load_df_results_from_hd(path_experiment)

            # calculate test metrics
            list_auc_test, list_accuracy_test, list_recall_test, list_precision_test = workflow_caroline.calculate_test_metrics(df_results, X_test, y_test)

            for idx_model in range(len(df_results['Model'])):

                model = df_results['Model'][idx_model]

                # get parameter names
                parameters = path_experiment.split(os.sep)[12:]
                filename = ""
                for p in parameters:
                    filename += "_" + p
                filename += "_" + model

                # fill in table
                list_results_test.append({'Parameter': filename,
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

        bla = 1

        #df_sorted.set_index('Parameter', inplace=True)
        #df_sorted.plot(kind='bar', legend=False)
        #plt.xlabel('Parameter')
        #plt.ylabel('Accuracy')
        #plt.title('Accuracy by Parameter')
        #plt.show()