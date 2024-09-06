import os
from lib.data_handler import hd
import settings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

if __name__ == '__main__':
    # for all machine learning folder:
    for FOLDER in settings.FEATURE_SET_LIST:

        #########################################################
        # define paths and load data
        print(FOLDER)

        SOURCE_DATA_FOLDER = FOLDER.replace("0_feature-set", "1_ML")

        base_folder = os.path.join(settings.PATH_RESULTS_FOLDER, SOURCE_DATA_FOLDER)

        full_path = os.path.join(base_folder, 'results_test.csv')

        df = hd.load_csv_as_df(full_path)

        ######################################################
        # Plot

        sns.set_palette("flare")  # crest


        # Define the parameters you want to plot and their desired orders
        parameter_orders = {
            "Window size": sorted(df["Window size"].unique()),  # Sorting the unique values
            "Bin size": sorted(df["Bin size"].unique()),  # Sorting the unique values
            "Window overlap": sorted(df["Window overlap"].unique()),  # Sorting the unique values
            "Correlation method": settings.CONNECTIVITY_METHODS
        }

        ml_models = settings.ML_MODELS

        # Create subplots
        fig, axes = plt.subplots(1, len(parameter_orders), figsize=(18, 6), sharey=True)

        # Plot each parameter in a separate subplot
        for ax, param in zip(axes, parameter_orders.keys()):
            # Ensure the x-axis is sorted by setting the order in the barplot
            #sns.barplot(x=param, y="AUC", hue="ML model", data=df, ax=ax,
            #            order=parameter_orders[param], hue_order=ml_models)
            sns.stripplot(x=param, y="AUC", hue="ML model", data=df, ax=ax,
                          order=parameter_orders[param], hue_order=ml_models, jitter=True, dodge=True,
                          palette="viridis")

            ax.set_title(f"AUC by {param}")
            ax.set_xlabel(param)
            ax.set_ylabel("AUC")

            # Set y-axis ticks from 0 to 1 with 0.1 steps
            ax.set_yticks(np.arange(0, 1.1, 0.1))

            legend = ax.legend(loc='lower center', frameon=True)
            legend.get_frame().set_edgecolor('black')  # Set the border color
            legend.get_frame().set_facecolor('white')  # Set the background color
            legend.get_frame().set_linewidth(1.5)

        # Adjust the layout
        plt.tight_layout()
        #plt.show()

        # save figure
        full_path = os.path.join(base_folder, "results_test.pdf")
        hd.save_figure(fig, full_path)


