# coding=utf-8

"""
Goal: Downloading financial data (related to stock markets) from diverse sources
      (Alpha Vantage, Yahoo Finance).
Authors: Thibaut Théate and Damien Ernst
Institution: University of Liège
"""

###############################################################################
################################### Imports ###################################
###############################################################################

import pandas as pd
import pandas_datareader as pdr
import requests
import yfinance as yf
import os

from io import StringIO

# Import pandas_ta for technical indicators
import pandas_ta as ta

###############################################################################
############################## Class AlphaVantage #############################
###############################################################################

class AlphaVantage:
    """
    GOAL: Downloading stock market data from the Alpha Vantage API. See the
          AlphaVantage documentation for more information.
    
    VARIABLES:  - link: Link to the Alpha Vantage website.
                - apikey: Key required to access the Alpha Vantage API.
                - datatype: 'csv' or 'json' data format.
                - outputsize: 'full' or 'compact' (only 100 time steps).
                - data: Pandas dataframe containing the stock market data.
                                
    METHODS:    - __init__: Object constructor initializing some variables.
                - getDailyData: Retrieve daily stock market data.
                - getIntradayData: Retrieve intraday stock market data.
                - processDataframe: Process the dataframe to homogenize the format.
    """

    def __init__(self):
        """
        GOAL: Object constructor initializing the class variables. 
        
        INPUTS: /      
        
        OUTPUTS: /
        """
        
        self.link = 'https://www.alphavantage.co/query'
        self.apikey = 'APIKEY'
        self.datatype = 'csv'
        self.outputsize = 'full'
        self.data = pd.DataFrame()
        
        
    def getDailyData(self, marketSymbol, startingDate, endingDate):
        """
        GOAL: Downloading daily stock market data from the Alpha Vantage API. 
        
        INPUTS:     - marketSymbol: Stock market symbol.
                    - startingDate: Beginning of the trading horizon.
                    - endingDate: Ending of the trading horizon.
          
        OUTPUTS:    - data: Pandas dataframe containing the stock market data.
        """
        
        # Send an HTTP request to the Alpha Vantage API
        payload = {'function': 'TIME_SERIES_DAILY_ADJUSTED', 'symbol': marketSymbol, 
                   'outputsize': self.outputsize, 'datatype': self.datatype, 
                   'apikey': self.apikey}
        response = requests.get(self.link, params=payload)
        
        # Process the CSV file retrieved
        csvText = StringIO(response.text)
        data = pd.read_csv(csvText, index_col='timestamp')
        
        # Process the dataframe to homogenize the output format
        self.data = self.processDataframe(data)
        if(startingDate != 0 and endingDate != 0):
            self.data = self.data.loc[startingDate:endingDate]

        return self.data
        
        
    def getIntradayData(self, marketSymbol, startingDate, endingDate, timePeriod=60):
        """
        GOAL: Downloading intraday stock market data from the Alpha Vantage API. 
        
        INPUTS:     - marketSymbol: Stock market symbol. 
                    - startingDate: Beginning of the trading horizon.
                    - endingDate: Ending of the trading horizon.
                    - timePeriod: Time step of the stock market data (in seconds).
          
        OUTPUTS:    - data: Pandas dataframe containing the stock market data.
        """
        
        # Round the timePeriod value to the closest accepted value
        possiblePeriods = [1, 5, 15, 30, 60]
        timePeriod = min(possiblePeriods, key=lambda x:abs(x-timePeriod))
        
        # Send a HTTP request to the AlphaVantage API
        payload = {'function': 'TIME_SERIES_INTRADAY', 'symbol': marketSymbol, 
                   'outputsize': self.outputsize, 'datatype': self.datatype, 
                   'apikey': self.apikey, 'interval': str(timePeriod)+'min'}
        response = requests.get(self.link, params=payload)
        
        # Process the CSV file retrieved
        csvText = StringIO(response.text)
        data = pd.read_csv(csvText, index_col='timestamp')
        
        # Process the dataframe to homogenize the output format
        self.data = self.processDataframe(data)
        if(startingDate != 0 and endingDate != 0):
            self.data = self.data.loc[startingDate:endingDate]

        return self.data
    
    
    def processDataframe(self, dataframe):
        """
        GOAL: Process a downloaded dataframe to homogenize the output format.
        
        INPUTS:     - dataframe: Pandas dataframe to be processed.
          
        OUTPUTS:    - dataframe: Processed Pandas dataframe.
        """
        
        # Reverse the order of the dataframe (chronological order)
        dataframe = dataframe[::-1]

        # Remove useless columns
        dataframe['close'] = dataframe['adjusted_close']
        del dataframe['adjusted_close']
        del dataframe['dividend_amount']
        del dataframe['split_coefficient']
        
        # Adapt the dataframe index and column names
        dataframe.index.names = ['Timestamp']
        dataframe = dataframe.rename(index=str, columns={"open": "Open",
                                                         "high": "High", 
                                                         "low": "Low",
                                                         "close": "Close",
                                                         "volume": "Volume"})
        # Adjust the format of the index values
        dataframe.index = dataframe.index.map(pd.Timestamp)

        return dataframe



###############################################################################
########################### Class YahooFinance ################################
###############################################################################

class YahooFinance:   
    """
    GOAL: Downloading stock market data from the Yahoo Finance API using yfinance.
        
    VARIABLES:  - data: Pandas dataframe containing the stock market data.
                                
    METHODS:    - __init__: Object constructor initializing some variables.
                - getDailyData: Retrieve daily stock market data.
                - processDataframe: Process a dataframe to homogenize the
                                    output format.
    """
        
    def __init__(self):
        """
        GOAL: Object constructor initializing the class variables. 
        """
        self.data = pd.DataFrame()

    def getDailyData(self, marketSymbol, startingDate, endingDate):
        """
        GOAL: Downloading daily stock market data from Yahoo Finance using yfinance. 
            
        INPUTS:     - marketSymbol: Stock market symbol.
                    - startingDate: Beginning of the trading horizon.
                    - endingDate: Ending of the trading horizon.
              
        OUTPUTS:    - data: Pandas dataframe containing the stock market data.
        """
        # Ensure marketSymbol is a string, not a list
        if isinstance(marketSymbol, list) and len(marketSymbol) == 1:
            marketSymbol = marketSymbol[0]
        
        data = yf.download(marketSymbol, start=startingDate, end=endingDate)
        self.data = self.processDataframe(data)
        return self.data

    def processDataframe(self, dataframe):
        """
        GOAL: Process a downloaded dataframe to homogenize the output format.
            
        INPUTS:     - dataframe: Pandas dataframe to be processed.
              
        OUTPUTS:    - dataframe: Processed Pandas dataframe.
        """
        # If columns are multi-level, flatten them
        if isinstance(dataframe.columns, pd.MultiIndex):
            # Flatten the columns by taking the first level
            dataframe.columns = dataframe.columns.get_level_values(0)
        
        # Compute the adjustment factor
        adj_factor = dataframe['Adj Close'] / dataframe['Close']

        # Adjust the 'Open', 'High', and 'Low' prices
        dataframe['Open'] = dataframe['Open'] * adj_factor
        dataframe['High'] = dataframe['High'] * adj_factor
        dataframe['Low'] = dataframe['Low'] * adj_factor
        dataframe['Close'] = dataframe['Adj Close']

        # Remove 'Adj Close' column
        del dataframe['Adj Close']

        # Adapt the dataframe index and column names
        dataframe.index.names = ['Timestamp']
        dataframe = dataframe[['Open', 'High', 'Low', 'Close', 'Volume']]

        # Add technical indicators
        dataframe = self.addTechnicalIndicators(dataframe)

        return dataframe

    def addTechnicalIndicators(self, dataframe):
        """
        GOAL: Add technical indicators to the dataframe.

        INPUTS:     - dataframe: Pandas dataframe with stock data.

        OUTPUTS:    - dataframe: Pandas dataframe with technical indicators added.
        """
        # Add Simple Moving Averages (SMA)
        dataframe['SMA_10'] = ta.sma(dataframe['Close'], length=10)
        dataframe['SMA_20'] = ta.sma(dataframe['Close'], length=20)

        # Add Exponential Moving Averages (EMA)
        dataframe['EMA_10'] = ta.ema(dataframe['Close'], length=10)
        dataframe['EMA_20'] = ta.ema(dataframe['Close'], length=20)

        # Add Relative Strength Index (RSI)
        dataframe['RSI_14'] = ta.rsi(dataframe['Close'], length=14)

        # Add Moving Average Convergence Divergence (MACD)
        macd = ta.macd(dataframe['Close'])
        dataframe['MACD'] = macd['MACD_12_26_9']
        dataframe['MACD_Signal'] = macd['MACDs_12_26_9']
        dataframe['MACD_Hist'] = macd['MACDh_12_26_9']

        # Add Bollinger Bands
        bollinger = ta.bbands(dataframe['Close'], length=20, std=2)
        dataframe['BB_Middle'] = bollinger['BBM_20_2.0']
        dataframe['BB_Upper'] = bollinger['BBU_20_2.0']
        dataframe['BB_Lower'] = bollinger['BBL_20_2.0']

        # Add Average True Range (ATR)
        dataframe['ATR_14'] = ta.atr(dataframe['High'], dataframe['Low'], dataframe['Close'], length=14)

        # Add On-Balance Volume (OBV)
        dataframe['OBV'] = ta.obv(dataframe['Close'], dataframe['Volume'])

        # Fill any NaN values that may have been introduced by the indicators
        dataframe.fillna(method='ffill', inplace=True)
        dataframe.fillna(method='bfill', inplace=True)

        return dataframe


    
###############################################################################
############################# Class CSVHandler ################################
###############################################################################
    
class CSVHandler:
    """
    GOAL: Converting "Pandas dataframe" <-> "CSV file" (bidirectional).
        
    METHODS:    - dataframeToCSV: Saving a dataframe into a CSV file.
                - CSVToDataframe: Loading a CSV file into a dataframe.
    """
        
    def dataframeToCSV(self, name, dataframe):
        """
        GOAL: Saving a dataframe into a CSV file.
            
        INPUTS:     - name: Name of the CSV file.   
                    - dataframe: Pandas dataframe to be saved.
              
        OUTPUTS: /
        """
        # Extract directory path and create if it doesn't exist
        directory = os.path.dirname(name)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)  # Create missing directories

        path = name + '.csv'
        dataframe.to_csv(path, index_label='Timestamp')
            
    def CSVToDataframe(self, name):
        """
        GOAL: Loading a CSV file into a dataframe.
            
        INPUTS:     - name: Name of the CSV file.   
              
        OUTPUTS:    - dataframe: Pandas dataframe loaded.
        """
        path = name + '.csv'
        return pd.read_csv(path,
                           header=0,
                           index_col='Timestamp',
                           parse_dates=True)