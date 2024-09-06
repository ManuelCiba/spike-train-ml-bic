# TODO: statistical tests: omit nan, MI: replaced nan by zero !!!!

# TODO: What could be improved compared to the last paper version:
# TODO: complex network measures: threshold 0.5 -> 2*std()+mean()
# TODO: not only use complex network measures + synchrony value but all (matrices + measures + synchrony curves)
# TODO: when flatting matrices: is it correct to replace nan by zeros?

# Author: Manuel Ciba, 2024
# The machine learning script and complex network measure calculation is based on different works from
# Caroline Lourenco Alves.
# If using this workflow, please cite the following works:
# [1] Alves, Caroline L., et al. "EEG functional connectivity and deep learning for automatic diagnosis of brain disorders: Alzheimer’s disease and schizophrenia." Journal of Physics: complexity 3.2 (2022): 025001.
# [2] Alves, Caroline L., et al. "Diagnosis of autism spectrum disorder based on functional brain networks and machine learning." Scientific Reports 13.1 (2023): 8072.
# [3] Alves, Caroline L., et al. "Application of machine learning and complex network measures to an EEG dataset from ayahuasca experiments." Plos one 17.12 (2022): e0277257.
# [4] Alves, Caroline L., et al. "On the advances in machine learning and complex network measures to an EEG dataset from DMT experiments." Journal of Physics: Complexity 5.1 (2024): 015002.
# [5] Alves, Caroline L., et al. "Analysis of functional connectivity using machine learning and deep learning in different data modalities from individuals with schizophrenia." Journal of Neural Engineering 20.5 (2023): 056025.

# How to produce requirements.txt:
# In pycharm, go to terminal and enter:
# pip freeze > requirements.txt

import time
import os
import numpy as np
import warnings

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    time_list = []

    time_list.append(time.time())

    exec(open(os.path.join(os.getcwd(), "script0_1_calculate_spiketrain_features.py")).read())
    time_list.append(time.time())
    exec(open(os.path.join(os.getcwd(), "script0_2_calculate_correlation_matrices_raw.py")).read())
    time_list.append(time.time())
    exec(open(os.path.join(os.getcwd(), "script1_scale_matrices.py")).read())
    time_list.append(time.time())
    exec(open(os.
              path.join(os.getcwd(), "script1_scale_matrices_independent.py")).read())
    time_list.append(time.time())
    exec(open(os.path.join(os.getcwd(), "script2_calculate_complex_network_measures.py")).read())
    time_list.append(time.time())
    exec(open(os.path.join(os.getcwd(), "script2_calculate_complex_network_measures_independent.py")).read())
    time_list.append(time.time())
    exec(open(os.path.join(os.getcwd(), "script3_0_collect_features.py")).read())
    time_list.append(time.time())
    exec(open(os.path.join(os.getcwd(), "script3_1_machine_learning.py")).read())
    time_list.append(time.time())
    exec(open(os.path.join(os.getcwd(), "script4_plot_features.py")).read())
    time_list.append(time.time())
    exec(open(os.path.join(os.getcwd(), "script4_find_best_result.py")).read())
    time_list.append(time.time())
    exec(open(os.path.join(os.getcwd(), "script5_plot_ml_results.py")).read())
    time_list.append(time.time())


    duration_seconds = np.diff(np.array(time_list))
    duration_minutes = duration_seconds / 60
    print("Duration each script (in minutes): " + str(duration_minutes))
    print("Duration all scripts (in minutes): " + str(np.sum(duration_minutes)))


