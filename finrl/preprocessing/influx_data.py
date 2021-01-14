"""
Created on Fri Aug  7 05:33:56 2020

@author: Wong Wei Jie
"""

from influxdb import InfluxDBClient
from datetime import datetime, timedelta
from finrl.config import config
import pandas as pd
from matplotlib.dates import date2num, DateFormatter
import matplotlib.pyplot as plt
import pytz

client = InfluxDBClient(host='localhost', port=8086)


def load_from_influx_chicago(currency_pair: str, window_period: int): 
    """load single currency pair tick data from Influx db

    Args:
        currency_pair (str) : for e.g. "EURUSD" 
        window_period (int) : from a window_period seconds before an event trigger until a 
                            window_period seconds after. window_frame is twice that of the
                            window_period.

    Returns:
        list of pandas dataframe
    """

    chicago_pmi = pd.read_csv(f"{config.REF_DATA_SAVE_DIR}/Chicago_PMI_releases.csv", header=None)
    for index, row in chicago_pmi.iterrows():
        eastern_tz = pytz.timezone('Asia/Singapore')
        current_day = datetime.strptime(row[0][:12], '%b %d, %Y')
        current_day = eastern_tz.localize(current_day)
        time_array = row[1].split(":")
        local_time = timedelta(hours = int(time_array[0]) + 12, minutes = int(time_array[1]))
        news_time = current_day + local_time
        print(news_time.strftime("%d/%m/%Y, %H:%M:%S%z"))
        timestamp = news_time.timestamp()
        print(timestamp)
        chicago_pmi.iloc[index, 5] = timestamp
        
    for _, event_timestamp in chicago_pmi.loc[(chicago_pmi[5]<1577836800) & (chicago_pmi[5]>1356998400),5].iteritems():
        lower_limit_timestamp = event_timestamp - window_period
        upper_limit_timestamp = event_timestamp + window_period

        
        query_statement = 'select "bid", "ask", "bid_vol", "ask_vol" from "dukascopy"."autogen"."EURUSD" where time >= ' +\
        str(int(lower_limit_timestamp)) + '000000000' +\
        ' and time <= ' + str(int(upper_limit_timestamp)) + '000000000 order by time asc' 
        
        
        results = client.query(query_statement)
        time_values = []
        bid_values = []
        ask_values = []
        
        df = pd.DataFrame(columns = ["time", "bid", "ask", "bid_vol", "ask_vol"])
        
        for point in results.get_points(): 
            
            try : 
                date_time_value = datetime.strptime(point["time"], '%Y-%m-%dT%H:%M:%S.%fZ')
            except : 
                date_time_value = datetime.strptime(point["time"], '%Y-%m-%dT%H:%M:%SZ')
                
            time_values.append(date_time_value)
            bid_values.append(point["bid"])
            ask_values.append(point["ask"])
            
    
            row = {"time" : date_time_value,
                "bid" : point["bid"], 
                "ask" : point["ask"],
                "bid_vol" : point["bid_vol"],
                "ask_vol" : point["ask_vol"],
                }
            
            df = df.append(row, ignore_index = True)
            
        
        
    return df

def load_from_influx_query(currency_pair : str, start_date_time : str, end_date_time : str):
    """load currency pair tick data from influx through a timeframe query

    Args:
        currency_pair (str) : for e.g. "EURUSD" 
        start_date_time (int) : for e.g. "01-01-2020 13:45:00" in UTC time
        end_date_time (int) : for e.g. "01-01-2020 13:45:00" in UTC time

    
    Returns:
         pandas dataframe 
    """
    start_timestamp = datetime.strptime(start_date_time, "%d-%m-%Y %H:%M:%S").timestamp()
    end_timestamp = datetime.strptime(end_date_time, "%d-%m-%Y %H:%M:%S").timestamp()

    query_statement = 'select "bid", "ask", "bid_vol", "ask_vol" from "dukascopy"."autogen"."EURUSD" where time >= ' +\
        str(int(start_timestamp)) + '000000000' +\
        ' and time <= ' + str(int(end_timestamp)) + '000000000 order by time asc'

    results = client.query(query_statement)

    df = pd.DataFrame(columns = ["time", "bid", "ask", "bid_vol", "ask_vol"])

    for point in results.get_points(): 
            
            try : 
                date_time_value = datetime.strptime(point["time"], '%Y-%m-%dT%H:%M:%S.%fZ')
            except : 
                date_time_value = datetime.strptime(point["time"], '%Y-%m-%dT%H:%M:%SZ')
    
            row = {"time" : date_time_value,
                "bid" : point["bid"], 
                "ask" : point["ask"],
                "bid_vol" : point["bid_vol"],
                "ask_vol" : point["ask_vol"],
                }
            
            df = df.append(row, ignore_index = True)

    return df

def export_csv_from_influx_chicago(currency_pair: str, window_period: int, dir_path: str) :
    """exports as csv a single currency pair tick data from Influx db between a certain
        window period before and after Chicago PMI

    Args:
        currency_pair (str) : for e.g. "EURUSD" 
        window_period (int) : from a window_period seconds before an event trigger until a 
                            window_period seconds after. window_frame is twice that of the
                            window_period.
        dir_path (str) : specify a dir path from project root to export csv to. for e.g. 
                            finrl/preprocessing/datasets/chicago_pmi/EURUSD/raw

    Returns:
        None
    """ 

    chicago_pmi = pd.read_csv(f"{config.REF_DATA_SAVE_DIR}/Chicago_PMI_releases.csv", header=None)
    for index, row in chicago_pmi.iterrows():
        eastern_tz = pytz.timezone('Asia/Singapore')
        current_day = datetime.strptime(row[0][:12], '%b %d, %Y')
        current_day = eastern_tz.localize(current_day)
        time_array = row[1].split(":")
        local_time = timedelta(hours = int(time_array[0]) + 12, minutes = int(time_array[1]))
        news_time = current_day + local_time
        print(news_time.strftime("%d/%m/%Y, %H:%M:%S%z"))
        timestamp = news_time.timestamp()
        print(timestamp)
        chicago_pmi.iloc[index, 5] = timestamp
        
    for _, event_timestamp in chicago_pmi.loc[(chicago_pmi[5]<1577836800) & (chicago_pmi[5]>1356998400),5].iteritems():
        lower_limit_timestamp = event_timestamp - window_period
        upper_limit_timestamp = event_timestamp + window_period

        
        query_statement = f'select "bid", "ask", "bid_vol", "ask_vol" from "dukascopy"."{currency_pair}" where time >= ' +\
        str(int(lower_limit_timestamp)) + '000000000' +\
        ' and time <= ' + str(int(upper_limit_timestamp)) + '000000000 order by time asc' 
        
        
        results = client.query(query_statement)
        time_values = []
        bid_values = []
        ask_values = []
        
        df = pd.DataFrame(columns = ["time", "bid", "ask", "bid_vol", "ask_vol"])
        
        for point in results.get_points(): 
            
            try : 
                date_time_value = datetime.strptime(point["time"], '%Y-%m-%dT%H:%M:%S.%fZ')
            except : 
                date_time_value = datetime.strptime(point["time"], '%Y-%m-%dT%H:%M:%SZ')
                
            time_values.append(date_time_value)
            bid_values.append(point["bid"])
            ask_values.append(point["ask"])
            
    
            row = {"time" : point["time"],
                "bid" : point["bid"], 
                "ask" : point["ask"],
                "bid_vol" : point["bid_vol"],
                "ask_vol" : point["ask_vol"],
                }
            
            df = df.append(row, ignore_index = True)
        
        
        df.to_csv(f'{config.DATASET_DIR}/{dir_path}/{currency_pair}_Chicago_Pmi_' + datetime.fromtimestamp(event_timestamp).strftime('%Y-%m-%d') + ".csv", index = False)    

    return None