import os
import pandas as pd

def load_csv_as_df(full_path, index_col=False):
    # set index_col=0 if the csv file contains row-names
    # header=0 and index_col=0 will use the row and col names like in the csv file
    df_matrix = pd.read_csv(full_path, header=0, index_col=index_col)
    return df_matrix

def save_df_as_csv(df, full_path, index=False):
    # create directory
    result_folder = os.path.dirname(full_path)
    os.makedirs(result_folder, exist_ok=True)

    #df = pd.DataFrame(result)
    #df.to_csv(full_path, sep=',', header=header, index=False)
    df.to_csv(full_path, sep=',', index=index)

def save_figure(fig, full_path):
    # create directory
    result_folder = os.path.dirname(full_path)
    os.makedirs(result_folder, exist_ok=True)

    fig.savefig(full_path, bbox_inches='tight', dpi=300)
    print("    Saved: " + full_path)