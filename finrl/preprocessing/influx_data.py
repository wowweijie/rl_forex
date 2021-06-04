"""
Created on Fri Aug  7 05:33:56 2020

@author: Wong Wei Jie
"""

from influxdb import DataFrameClient, InfluxDBClient
from datetime import datetime, timedelta
from finrl.config import config
from finrl.preprocessing.data import export_dataset
import pandas as pd
from matplotlib.dates import date2num, DateFormatter
import matplotlib.pyplot as plt
import pytz

client = DataFrameClient(host='localhost', port=8086)

def load_from_influx_query(currency_pair : str, start_date_time : str, end_date_time : str):
    """load currency pair tick data from influx through a timeframe query

    Args:
        currency_pair (str) : for e.g. "EURUSD" 
        start_date_time (int) : for e.g. "01-01-2020 13:45:00" in UTC time
        end_date_time (int) : for e.g. "01-01-2020 13:45:00" in UTC time

    
    Returns:
         pandas dataframe 
    """
    start_datetime = pytz.utc.localize(datetime.strptime(start_date_time, "%d-%m-%Y %H:%M:%S"))
    end_datetime = pytz.utc.localize(datetime.strptime(end_date_time, "%d-%m-%Y %H:%M:%S"))

    start_timestamp = start_datetime.timestamp()
    end_timestamp = end_datetime.timestamp()

    query_statement = f'select "Bid price", "Ask price", "Bid volume", "Ask volume" from "dukascopy".."{currency_pair}" where time >= ' +\
        str(int(start_timestamp)) + '000000000' +\
        ' and time <= ' + str(int(end_timestamp)) + '000000000 order by time asc'

    print(query_statement)

    results = client.query(query_statement)

    df = pd.DataFrame.from_dict(results[currency_pair], orient='columns')

    df.rename(columns={
        'Bid price' : 'bid',
        'Ask price' : 'ask',
        'Bid volume' : 'bid_vol',
        'Ask volume' : 'ask_vol'
    }, inplace=True) 

    df = df.astype("float64")   

    return df

def export_csv_aggregated_from_influx(currency_pair : str, start_date_time : str, end_date_time : str, ohlc_interval: str):
    """export a continuous window of ohlc data aggregated in a specified time interval to specified file path

    Args:
        currency_pair (str) : for e.g. "EURUSD" 
        start_date_time (int) : for e.g. "01-01-2020 13:45:00" in UTC time
        end_date_time (int) : for e.g. "01-01-2020 13:45:00" in UTC time
        file_path (str) : from finrl/preprocessing/datasets, e.g. "15min/EURUSD/01_19.csv" 

    Returns:
        None. CSV file saved in file path target location
    """

    df = load_from_influx_query(currency_pair, start_date_time, end_date_time)

    # setting index to datetime
    # df['time'] = pd.to_datetime(df['time'], yearfirst = True)
    # df.set_index('time', inplace=True)

    # setting count
    df['tick_count'] = 0

    df = df.resample(ohlc_interval).agg({'ask':'ohlc','bid':'ohlc','bid_vol':'sum','ask_vol':'sum', 'tick_count' : 'count'})
    fillna_values = dict.fromkeys((('ask', col) for col in df['ask'].columns.tolist()),df['ask']['close'].ffill())
    fillna_values.update(dict.fromkeys((('bid', col) for col in df['bid'].columns.tolist()),df['bid']['close'].ffill()))
    df.fillna(fillna_values, inplace = True)

    return df