from __future__ import division,absolute_import,print_function
from math import ceil
from config import config
import numpy as np
import pandas as pd


def load_dataset(file_path: str):
    """
    load csv dataset from file path 

    Args:
        file_path (str) : in context/currency_pair/filename format for e.g. "chicago_pmi/EURUSD/filename" 


    Returns:
        pandas dataframe
    """
    _data = pd.read_csv(f"preprocessing/datasets/{file_path}")
    _data['time'] = pd.to_datetime(_data['time'], yearfirst = True)
    _data.set_index('time', inplace=True)
    return _data

def train_test_split(df, train_fraction):
    """
    splits time-indexed dataframe into train and test

    Args:
        train_fraction (float) : portion of data to be trainset"


    Returns:
        (train_df, test_df) a tuple of trainset and testset dataframe
    """
    last_index = ceil(len(df) * train_fraction) - 1 
    train_df = df.loc[df.index[0] : df.index[last_index]]
    test_df = df.loc[df.index[last_index + 1] : ]
    return train_df, test_df


def convert_to_datetime(time):
    time_fmt = '%Y-%m-%dT%H:%M:%S'
    if isinstance(time, str):
        return datetime.datetime.strptime(time, time_fmt)




