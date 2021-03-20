import numpy as np
import pandas as pd
import copy
from finrl.config import config
from talib import abstract


class FeatureEngineer:
    """Provides methods for preprocessing the stock price data

    Attributes
    ----------
        df: DataFrame
            data imported from ohlc folders of currency pairs
        use_technical_indicator : boolean
            we technical indicator or not
        tech_indicator_list : list
            a list of technical indicator names (modified from config.py)
        tech_indicator_params_map = A nested dictionary to describe
        

    Methods
    -------
    preprocess_data()
        main method to do the feature engineering

    """
    def __init__(self, 
        df,
        use_technical_indicator=True,
        user_defined_feature = True,
        tech_indicator_list = config.TECHNICAL_INDICATORS_LIST,
        tech_indicator_params_map = {
                'SMA': {'time_period' : 20}, #time_period in seoonds
                'EMA': {'time_period' : 20}, #time_period in seoonds
            } 
        ):

        self.df = df
        self.use_technical_indicator = use_technical_indicator
        self.user_defined_feature = user_defined_feature
        self.tech_indicator_list = tech_indicator_list
        self.tech_indicator_parameter_map = copy.deepcopy(tech_indicator_params_map)

        #type_list = self._get_type_list(5)
        #self.__features = type_list
        #self.__data_columns = config.DEFAULT_DATA_COLUMNS + self.__features

    def format_ohlc(self, df):
        """
        Format tick-level timeseries data into OHLC

        Returns:
            pandas dataframe of timeseries in OHLC 1 second format
        """
        df = df.resample('1S').agg({'ask':'ohlc','bid':'ohlc','bid_vol':'sum','ask_vol':'sum'})
        fillna_values = dict.fromkeys((('ask', col) for col in df['ask'].columns.tolist()),df['ask']['close'].ffill())
        fillna_values.update(dict.fromkeys((('bid', col) for col in df['bid'].columns.tolist()),df['bid']['close'].ffill()))
        df=df.fillna(fillna_values)
        return df


    def preprocess_data(self):
        """
        Main method that does feature engineering. Firstly, it creates a new sub dataframe called "ovr" that
        derives and average price from the OHLC of bid and ask, while summing the volume of bid and ask. If 

        Returns:
            pandas dataframe of OHLC dataframe with technical indicators and generated columns
        """
        self.df = self.df.dropna()

        self.df['ovr', 'open'] = self.df.apply(lambda row: (row['ask']['open'] + row['bid']['open'])/2, axis = 1)
        self.df['ovr', 'high'] = self.df.apply(lambda row: (row['ask']['high'] + row['bid']['high'])/2, axis = 1)
        self.df['ovr', 'low'] = self.df.apply(lambda row: (row['ask']['low'] + row['bid']['low'])/2, axis = 1)
        self.df['ovr', 'close'] = self.df.apply(lambda row: (row['ask']['close'] + row['bid']['close'])/2, axis = 1)
        self.df['ovr', 'volume'] = self.df.apply(lambda row: row['bid_vol']['bid_vol'] + row['ask_vol']['ask_vol'], axis = 1)
        
        # add tic 
        self.df.loc[:, ('ovr', 'tic')] = 'FX'

        # add technical indicators
        # stockstats require all 5 columns
        if (self.use_technical_indicator==True):
            
            # add technical indicators using stockstats
            self.add_technical_indicator()
            print("Successfully added technical indicators")

        # add user defined feature
        if self.user_defined_feature == True:
            self.df = self.add_user_defined_feature(self.df)
            print("Successfully added user defined features")


        return self.df


    def add_technical_indicator(self):
        """
        Adds technical indicators from TA-lib onto new columns of the OHLCV dataframe in self.df

        Returns:
            pandas dataframe of OHLCV dataframe with technical indicators and generated columns
        """
        
        for indicator in self.tech_indicator_list:
            try :
                params = self.tech_indicator_parameter_map.get(indicator)
                talib_name = params.pop('talib_name')
                
            except KeyError as err:
                print(f"tech_indicator {indicator} not added as it is not specified in parameter mapping, ", err) 
                continue
            
            try:
                ta_hook = abstract.Function(talib_name)
                new_features = ta_hook(self.df['ovr'], **params)

                if type(new_features) is pd.Series: 
                    self.df['ovr', indicator] =  new_features
                
                elif type(new_features) is pd.DataFrame:
                    for colname in list(new_features.columns):
                        modified_colname = indicator + "_" + colname
                        self.df.loc[:,('ovr', modified_colname)] =  new_features[colname]

            
            except Exception as err:
                print(f'{indicator} does not exist or has wrong parameters, ', err)
                continue
            


    def add_user_defined_feature(self, df):
        """
         add user defined features
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """          
        
        df.loc[:, ('ovr','daily_return')]=df['ovr','close'].pct_change(1)
        #df['return_lag_1']=df.close.pct_change(2)
        #df['return_lag_2']=df.close.pct_change(3)
        #df['return_lag_3']=df.close.pct_change(4)
        #df['return_lag_4']=df.close.pct_change(5)
        return df


    def add_turbulence(self, data):
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        turbulence_index = self.calcualte_turbulence(df)
        df = df.merge(turbulence_index, on='date')
        df = df.sort_values(['date','tic']).reset_index(drop=True)
        return df


    def calcualte_turbulence(self, data):
        """calculate turbulence index based on dow 30"""
        # can add other market assets
        df = data.copy()
        df_price_pivot=df.pivot(index='date', columns='tic', values='close')
        unique_date = df.date.unique()
        # start after a year
        start = 252
        turbulence_index = [0]*start
        #turbulence_index = [0]
        count=0
        for i in range(start,len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            hist_price = df_price_pivot[[n in unique_date[0:i] for n in df_price_pivot.index ]]
            cov_temp = hist_price.cov()
            current_temp=(current_price - np.mean(hist_price,axis=0))
            temp = current_temp.values.dot(np.linalg.inv(cov_temp)).dot(current_temp.values.T)
            if temp>0:
                count+=1
                if count>2:
                    turbulence_temp = temp[0][0]
                else:
                    #avoid large outlier because of the calculation just begins
                    turbulence_temp=0
            else:
                turbulence_temp=0
            turbulence_index.append(turbulence_temp)
        
        
        turbulence_index = pd.DataFrame({'date':df_price_pivot.index,
                                         'turbulence':turbulence_index})
        return turbulence_index

    def _get_type_list(self, feature_number):
        """
        :param feature_number: an int indicates the number of features
        :return: a list of features n
        """
        if feature_number == 1:
            type_list = ["close"]
        elif feature_number == 2:
            type_list = ["close", "volume"]
            #raise NotImplementedError("the feature volume is not supported currently")
        elif feature_number == 3:
            type_list = ["close", "high", "low"]
        elif feature_number == 4:
            type_list = ["close", "high", "low", "open"]
        elif feature_number == 5:
            type_list = ["close", "high", "low", "open","volume"]  
        else:
            raise ValueError("feature number could not be %s" % feature_number)
        return type_list