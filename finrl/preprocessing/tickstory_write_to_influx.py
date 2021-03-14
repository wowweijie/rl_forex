# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 00:51:55 2020

@author: wongweijie
"""

import pandas as pd
import strict_rfc3339 as rfc
from influxdb import InfluxDBClient

class InfluxDBManager():
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.client = None

    def __enter__(self):
        self.client = InfluxDBClient(
            host = self.host, port = self.port
        )
        return self.client

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.client.__exit__(exc_type, exc_value, exc_traceback)

def write_points_from_csv_to_influx(filename: str, symbol: str):
    #filename - e.g. "EURUSD_Ticks_06.01.2017-06.01.2017.csv" (with .csv extension)
    #symbol - instrument pair e.g. "EURUSD"

    chunksize = 10 ** 6

    for df in pd.read_csv(filename, chunksize=chunksize, header=0, dtype='str'):

        with InfluxDBManager(host='localhost', port=8086) as client:

            print("df.shape" , df.shape)
            print("df size", df.count())
            print(df.head())
            lines = [] 
            
            for _, row in df.iterrows() :
                bid = row["Bid price"]
                ask = row["Ask price"]
                bid_vol = row["Bid volume"]
                ask_vol = row["Ask volume"]
                time = row["Timestamp"]
                try: 
                    timestamp = str(rfc.rfc3339_to_timestamp(time[:4]+'-'+time[4:6]+'-'+time[6:8]+'T'+time[9:17]+"."+time[18:]+"000000Z"))
                    len_stamp = len(timestamp)
                    timestamp = timestamp[:10]+timestamp[11:len_stamp]+'0'*(20-len_stamp)
                    lines.append(symbol  + ",type="+symbol
                        + " "
                        + "bid=" + bid + ","
                        + "ask=" + ask + ","
                        + "bid_vol=" + bid_vol + ","
                        + "ask_vol=" + ask_vol + " "
                        + timestamp)

                except:
                    print(f"Error caught on {row}")
                    continue
            
            print(str(len(lines)) + " lines")

            ## comment out this portion to expedite write process ##
            # thefile = open(filename[:-4] + '.txt', 'x', newline='')
            
            # for item in lines:
            #     thefile.write(item+'\n')
                
            # thefile.close()
            ####################################################
            
            # write to influx
            client.write_points(lines, database='dukascopy', time_precision='n', batch_size=10000, protocol='line')
