# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 00:51:55 2020

@author: wongweijie
"""

import pandas as pd
import strict_rfc3339 as rfc
from influxdb import InfluxDBClient, DataFrameClient

def write_points_from_csv_to_influx(filename: str, symbol: str, chunksize: int):
    """
    Args:
        filename - e.g. "EURUSD_Ticks_06.01.2017-06.01.2017.csv" (with .csv extension)
        symbol - instrument pair e.g. "EURUSD". This will be the measurement in influxdb

    Returns:
        None
    """

    client = DataFrameClient(host='localhost', port=8086)

    for df in pd.read_csv(filename, chunksize=chunksize, header=0, dtype='str'):

        print("df.shape" , df.shape)
        print("df size", df.count())
        print(df.head())
        
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%Y%m%d %H:%M:%S:%f")
        df.set_index('Timestamp', inplace=True)
        client.write_points(df, time_precision='ms', measurement=symbol, database='dukascopy')
