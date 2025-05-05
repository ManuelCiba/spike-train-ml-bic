import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import os
import quantities as pq

from lib.data_handler import folder_structure
import settings
from lib.data_handler import hd

# Function to return star notation based on p-value
def get_star_annotation(p_value):
    if p_value < 0.0001:
        return '****'  # p < 0.0001
    elif p_value < 0.001:
        return '***'  # p < 0.001
    elif p_value < 0.01:
        return '**'  # p < 0.01
    elif p_value < 0.05:
        return '*'  # p < 0.05
    else:
        return 'ns'  # Not significant (p >= 0.05)

def calc_lmm(basepath, bic_n, bic):

    # Load the data
    #bic = pd.read_csv(r'bic10.csv', sep=',', header=0).drop(columns=['Diversity', 'Reciprocity'])
    #bic_n = pd.read_csv(r'bic00.csv', sep=',', header=0).drop(columns=['Diversity', 'Reciprocity'])

    # Add a Chip column to each dataframe
    # Assuming each chip has approximately an equal number of rows
    chip_ids = np.repeat(range(1, 10), len(bic) // 9)  # Repeat chip IDs for 9 chips
    bic['Chip'] = chip_ids
    bic_n['Chip'] = chip_ids

    # Create a list to store measures
    list_measures = []

    # Loop through the columns (features)
    for i in range(len(bic.columns) - 1):  # Excluding the 'Chip' column from the loop
        # Prepare the data for comparison
        n1 = pd.DataFrame(bic[bic.columns[i]].values, columns=['Measure'])
        n1['Condition'] = 'bic10'
        n1['Chip'] = bic['Chip']  # Add Chip information

        n2 = pd.DataFrame(bic_n[bic_n.columns[i]].values, columns=['Measure'])
        n2['Condition'] = 'bic00'
        n2['Chip'] = bic_n['Chip']  # Add Chip information

        # Concatenate bic10 and bic00 for comparison along the rows
        plot_subject = pd.concat([n2, n1], axis=0)
        list_measures.append(plot_subject)

    # Set seaborn style
    sns.set_style("white")
    f = plt.figure(figsize=(10, 10))
    features = bic.columns[:-1]  # Assuming both have the same columns (excluding 'Chip')
    counter = 0

    while counter < len(features):
        i = list_measures[counter].dropna()

        # Create violin plot
        ax = sns.violinplot(x="Condition", y="Measure", data=i, palette='viridis', inner=None)
        plt.setp(ax.collections, alpha=.4)

        # Overlay a small boxplot inside the violin plot
        sns.boxplot(x="Condition", y="Measure", data=i, width=0.2, boxprops={'zorder': 2, 'facecolor': 'lightgrey'},
                    showmeans=True,
                    meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black",
                               "markersize": "7"})
        sns.stripplot(x="Condition", y="Measure", data=i, jitter=True, color='grey', alpha=0.5, size=5)

        # Define the LMM formula (including the chip as a random effect)
        formula = f"Measure ~ Condition + (1|Chip)"

        # Fit the linear mixed model
        model = smf.mixedlm(formula, data=i, groups=i["Chip"])
        result = model.fit()

        # Extract p-value and annotate with star notation
        p_value = result.pvalues['Condition[T.bic10]']
        stars = get_star_annotation(p_value)

        # Get y-limits for positioning the line and star
        ylim = ax.get_ylim()
        y_line = ylim[1] * 0.96  # Position the line slightly below the top of the y-axis
        y_star = ylim[1] * 0.97  # Position the star just above the line

        # Draw a line connecting the two boxplots
        ax.plot([0, 1], [y_line, y_line], color='black', lw=1.5)

        # Add star annotation above the line
        ax.text(0.5, y_star, f'{stars}', ha='center', va='center', fontsize=18)

        # Set y-label and clean up plot
        ax.set_ylabel(features[counter], fontsize=18)

        # Save the plot
        #plt.savefig(basepath + "/lmm-chip/" + features[counter] + ".pdf", dpi=800, bbox_inches="tight")
        full_path = os.path.join(basepath, "lmm-chip", features[counter] + ".pdf")
        fig = plt.gcf()
        hd.save_figure(fig, full_path)
        plt.close()

        counter += 1

if __name__ == '__main__':

    # for all machine learning folder:
    for FOLDER in settings.FEATURE_SET_LIST:

        print(FOLDER)

        SOURCE_DATA_FOLDER = FOLDER

        # define parameter
        bin_sizes = [1 * pq.ms] #settings.BIN_SIZES
        window_sizes = [240 * pq.s] #settings.WINDOW_SIZES
        window_overlaps = [0.75] #settings.WINDOW_OVERLAPS
        methods = ["pearson"] #settings.CONNECTIVITY_METHODS
        groups = ["bic00", "bic10"]
        chip_names = folder_structure.get_all_chip_names()

        path_experiment_list = folder_structure.generate_paths(target_data_folder=SOURCE_DATA_FOLDER,
                                                               methods=methods,
                                                               bin_sizes=bin_sizes,
                                                               window_sizes=window_sizes,
                                                               window_overlaps=window_overlaps,
                                                               chip_names=[],
                                                               groups=[])

        # for all experiments (=different parameter)
        for path_experiment in path_experiment_list:

            # skip all feature sets that include matrices (=4096 features = 4096 plots)
            if ("03_0_feature-set_measures_synchrony-value" not in path_experiment and
                    "03_0_feature-set_measures_synchrony-value_independent" not in path_experiment):
                continue

            # load feature matrices
            full_path_bic00 = os.path.join(path_experiment, 'bic00.csv')
            full_path_bic10 = os.path.join(path_experiment, 'bic10.csv')
            matrix_df_list_bic00 = hd.load_csv_as_df(full_path_bic00, index_col=False)
            matrix_df_list_bic10 = hd.load_csv_as_df(full_path_bic10, index_col=False)

            calc_lmm(path_experiment, matrix_df_list_bic00, matrix_df_list_bic10)