import os

from lib.data_handler import folder_structure
import settings
from lib.data_handler import hd
from lib.plots.make_box_plot_for_two_groups import make_box_plot_for_two_groups


if __name__ == '__main__':
    # for all machine learning folder:
    for FOLDER in settings.FEATURE_SET_LIST:

        print(FOLDER)

        SOURCE_DATA_FOLDER = FOLDER

        base_folder = os.path.join(settings.PATH_RESULTS_FOLDER, SOURCE_DATA_FOLDER)
        path_experiment_list = folder_structure.get_all_paths(base_folder, 'overlap')

        # for all experiments (=different parameter)
        for path_experiment in path_experiment_list:

            # skip all feature sets that include matrices (=4096 features = 4096 plots)
            if "03_0_feature-set_measures_synchrony-value" not in path_experiment:
                continue

            # load feature matrices
            full_path_bic00 = os.path.join(path_experiment, 'bic00.csv')
            full_path_bic10 = os.path.join(path_experiment, 'bic10.csv')
            matrix_df_list_bic00 = hd.load_csv_as_df(full_path_bic00, index_col=False)
            matrix_df_list_bic10 = hd.load_csv_as_df(full_path_bic10, index_col=False)

            # Get the list of features (column names excluding the 'Group' column)
            features = matrix_df_list_bic00.columns[:]

            # Loop through each feature
            for feature in features:
                fig = make_box_plot_for_two_groups(matrix_df_list_bic00, matrix_df_list_bic10, "With", "Without", feature, "Bicuculline")

                # save figure
                full_path = os.path.join(path_experiment, feature + ".pdf")
                hd.save_figure(fig, full_path)

            # save plots (folder structure)
            #full_path = os.path.join(path_experiment, "result_train.pdf")
            #fig_train.savefig(full_path, bbox_inches='tight', dpi=3000)
            #full_path = os.path.join(path_experiment, "result_test.pdf")
            #fig_test.savefig(full_path, bbox_inches='tight', dpi=3000)

            # save plots (all in same folder)
            #parameters = path_experiment.split(os.sep)[12:]
            #filename = ""
            #for p in parameters:
            #    filename += "_" + p
            #full_path = os.path.join(base_folder, "train" + filename + ".pdf")
            #fig_train.savefig(full_path, bbox_inches='tight', dpi=3000)
            #full_path = os.path.join(base_folder, "test" + filename + ".pdf")
            #fig_test.savefig(full_path, bbox_inches='tight', dpi=3000)