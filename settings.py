import os
import quantities as pq

# KAI
PATH_RESULTS_FOLDER = os.path.join(os.getcwd(), "results")

# Macbook local
#PATH_RESULTS_FOLDER = "/home/mc/Documents/Data/Japan/results"
# Macbook Faubox
#PATH_RESULTS_FOLDER = "/home/mc/FAUbox/Work/Projects/2023-10_BIC_Japan/PYTHON/BIC-Japan_macBook/pythonProject/results"

# THAB
#PATH_RESULTS_FOLDER = "/home/manuel/Documents/Data/Japan/results"

# Folder names for the results
FOLDER_NAME_synchrony = "00_1_spiketrain_features"
FOLDER_NAME_matrices_raw = "00_correlation_matrices_raw"
FOLDER_NAME_matrices_scaled = "01_correlation_matrices_scaled"
FOLDER_NAME_matrices_scaled_independent = "01_correlation_matrices_scaled_independent"
FOLDER_NAME_measures = "02_complex_network_measures"
FOLDER_NAME_measures_independent = "02_complex_network_measures_independent"
FOLDER_NAME_features_matrices = "03_0_feature-set_matrices"
FOLDER_NAME_features_matrices_independent = "03_0_feature-set_matrices_independent"
FOLDER_NAME_features_measures = "03_0_feature-set_measures"
FOLDER_NAME_features_measures_independent = "03_0_feature-set_measures_independent"
FOLDER_NAME_features_measures_synchrony_value = "03_0_feature-set_measures_synchrony-value"
FOLDER_NAME_features_measures_synchrony_value_independent = "03_0_feature-set_measures_synchrony-value_independent"
FOLDER_NAME_features_measures_synchrony_curve = "03_0_feature-set_measures_synchrony-curve"
FOLDER_NAME_features_matrices_synchrony_curve = "03_0_feature-set_matrices_synchrony-curve"
FOLDER_NAME_features_synchrony_curve = "03_0_feature-set_synchrony-curve"
FOLDER_NAME_features_all = "03_0_feature-set_all"


# Parameter which will be calculated
BIN_SIZES = [1 * pq.ms, 5 * pq.ms]  #
WINDOW_SIZES = [10 * pq.s, 30 * pq.s, 60 * pq.s, 120 * pq.s, 240 * pq.s]  #  600 * pq.s
WINDOW_OVERLAPS = [0.2, 0.4, 0.5, 0]  # 0.5 0.5 = half window size
CONNECTIVITY_METHODS = ["tspe", "spearman", "canonical", "pearson"]
FEATURE_SET_LIST = [#FOLDER_NAME_features_matrices,
                    #FOLDER_NAME_features_measures,
                    FOLDER_NAME_features_measures_synchrony_value,
                    #FOLDER_NAME_features_measures_synchrony_curve,
                    #FOLDER_NAME_features_synchrony_curve,
                    #FOLDER_NAME_features_matrices_synchrony_curve
                    ]
ML_MODELS = ['RF', 'SVM', 'NB', 'KNN']  # excluded LR,  excluded 'XGboost' TAKES REALLY LONG AND 100% CPU