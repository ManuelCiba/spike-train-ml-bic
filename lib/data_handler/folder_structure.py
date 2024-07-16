import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np


def get_all_paths(base_folder, common_folder_name='rec'):
    # base_folder: Base folder where the results are stored

    # get all chip paths
    folder_all_chips_list = []
    for root, dirs, files in os.walk(base_folder):

        # Extract parameters from the folder path
        parameters = root.split(os.sep)[1:]  # Adjust the index based on your folder structure

        # Check if the folder matches the specified parameters
        try:
            index = parameters.index(next(s for s in parameters if common_folder_name in s))
            index += 1
        except:
            index = None
        else:
            if len(parameters) == index:
                folder_all_chips_list.append(root)

    # in case there are duplets, make the paths unique. here already unique
    folder_all_chips_list = list(set(folder_all_chips_list))
    folder_all_chips_list = sorted(folder_all_chips_list)
    return folder_all_chips_list