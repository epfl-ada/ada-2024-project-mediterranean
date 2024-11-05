# utils.py
# ==============================
# Title: ADA Project Data Processing Utilities
# ==============================

import pandas as pd
import os


def dataset_information(dataset, dataset_name):
    """Automatic dataset information retrieval

    Args:
        dataset (pandas.core.frame.DataFrame): pandas dataset
        dataset_name (str): name for output information

    Returns:
        None
    """

    print('\n')
    print('########################################################')
    print('We are starting analysing dataset', dataset_name)
    print('- Dimension of starting dataset:', dataset.shape)
    print('- Columns of dataset: ', dataset.columns)
    print('- Are all the id unique? Answer:', dataset[dataset.columns[0]].is_unique)
    print('- Are there some values that are NaN inside the dataset? Answer:',dataset.isna().any().any())
    print('Head: \n', dataset.head())
    return None

def ensure_col_types(dataset, cols_int, cols_float, cols_string):
    """_summary_

    Args:
        dataset (pandas.core.frame.DataFrame): _description_
        cols_num (str): _description_
        cols_string (str): _description_

    Returns:
        None
    """
    dataset[cols_int]  = dataset[cols_int].astype(int)
    dataset[cols_float]  = dataset[cols_float].astype(float)
    dataset[cols_string]  = dataset[cols_string].astype(str)
    return None


def write_csv_into_directory(output_dir, output_file, dataset):
    """_summary_

    Args:
        output_dir (_type_): _description_
        output_file (_type_): _description_
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, output_file)
    dataset.to_csv(file_path, index=False)
    print(f'Dataset successfully saved to {file_path}')
    return None

