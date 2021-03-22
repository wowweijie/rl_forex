from __future__ import division,absolute_import,print_function
from math import ceil
from finrl.config import config
import datetime
import numpy as np
import pandas as pd


def load_dataset(file_path: str):
    """
    load raw csv tick dataset from file path 

    Args:
        file_path (str) : from dataset folder in context/currency_pair/filename format for e.g. "chicago_pmi/EURUSD/ohlc/filename" 


    Returns:
        pandas dataframe
    """
    _data = pd.read_csv(f"{config.DATASET_DIR}/{file_path}")
    _data['time'] = pd.to_datetime(_data['time'], yearfirst = True)
    _data.set_index('time', inplace=True)
    return _data

def load_ohlc_dataset(file_path: str):
    return pd.read_csv(f"{config.DATASET_DIR}/{file_path}",
                 header = [0,1] ,index_col = 0,
                  parse_dates = True)

def export_dataset(df: pd.DataFrame, file_path: str):
    """
    exports csv dataset tp file path 

    Args:
        file_path (str) : from dataset folder in context/currency_pair/filename format for e.g. "chicago_pmi/EURUSD/ohlc/filename" 
        df (pd.Dataframe) : pandas dataframe


    Returns:
        pandas dataframe
    """
    df.to_csv(f"{config.DATASET_DIR}/{file_path}")
    print(f"dataset exported to {config.DATASET_DIR}/{file_path}")


def format_ohlc(df: pd.DataFrame, interval : str):
    """
    Format tick-level timeseries data

    Args:
        interval (str) : '1S' for 1 second ohlc, '15min' for 15 minute ohlc

    Returns:
        pandas dataframe of timeseries in OHLC 1 second format
    """
    df = df.resample('1S').agg({'ask':'ohlc','bid':'ohlc','bid_vol':'sum','ask_vol':'sum'})
    fillna_values = dict.fromkeys((('ask', col) for col in df['ask'].columns.tolist()),df['ask']['close'].ffill())
    fillna_values.update(dict.fromkeys((('bid', col) for col in df['bid'].columns.tolist()),df['bid']['close'].ffill()))
    df=df.fillna(fillna_values)
    return df

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

def data_split(df, start, end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df.date >= start) & (df.date < end)]
    data = data.sort_values(["date", "tic"], ignore_index=True)
    data.index = data.date.factorize()[0]
    return data



def convert_to_datetime(time):
    time_fmt = '%Y-%m-%dT%H:%M:%S'
    if isinstance(time, str):
        return datetime.datetime.strptime(time, time_fmt)




