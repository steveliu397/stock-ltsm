from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
# import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler

DATA_SOURCE = 'alphavantage'
API_KEY = '8NR1BD7HMN9AYF44'

class DataReader:
    ticker = "AAL"
    file_to_save = ''

    def __init__(self, ticker, file_to_save):
        self.ticker = ticker
        self.file_to_save = file_to_save

    def get_data(self):
        # use API as default method
        if DATA_SOURCE == "alphavantage":
            # configures URL and name of CSV file
            url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(self.ticker,API_KEY)
            
            with urllib.request.urlopen(url_string) as url:
                data = json.loads(url.read().decode())
                # extract stock market data
                data = data['Time Series (Daily)']
                df = pd.DataFrame(columns=['Date','Low','High','Close','Open'])
                for k,v in data.items():
                    date = dt.datetime.strptime(k, '%Y-%m-%d')
                    data_row = [date.date(),float(v['3. low']),float(v['2. high']),
                                float(v['4. close']),float(v['1. open'])]
                    df.loc[-1,:] = data_row
                    df.index = df.index + 1
            print('Data saved to : %s'%self.file_to_save)        
            df.to_csv(self.file_to_save)

        # move this to starter
        else:
            raise Exception('Cannot be found using Alpha Vantage')