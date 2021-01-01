from __future__ import division,absolute_import,print_function
import numpy as np
import pandas as pd


def load_dataset(file_path: str):
    """
    load csv dataset from file path 

    Args:`
        file_path (str) : in context/currency_pair/filename format for e.g. "chicago_pmi/EURUSD/filename" 


    Returns:
        pandas dataframe
    """
    _data = pd.read_csv("datasets/{file_path}")
    return _data

def data_split(df,start,end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df.date >= start) & (df.date < end)]
    data=data.sort_values(['date','tic'],ignore_index=True)
    data.index = data.date.factorize()[0]
    return data


def convert_to_datetime(time):
    time_fmt = '%Y-%m-%dT%H:%M:%S'
    if isinstance(time, str):
        return datetime.datetime.strptime(time, time_fmt)


