from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
# import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler

class DataReader:
    ticker = "AAL"

    def __init__(self, ticker):
        self.ticker = ticker

    def get_data(self):
        # use API as default method
        data_source = "alphavantage"
        api_key = '8NR1BD7HMN9AYF44'

        if data_source == "alphavantage":
            # configures URL and name of CSV file
            url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(self.ticker,api_key)
            file_to_save = 'stock_market_data-%s.csv'%self.ticker
            
            if not os.path.exists(file_to_save):
                # load from URL
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
                print('Data saved to : %s'%file_to_save)        
                df.to_csv(file_to_save)

            else:
                print('File already exists. Loading data from CSV')
                df = pd.read_csv(file_to_save)

        else:
            # read directly from saved files in the Stocks folder
            df = pd.read_csv(os.path.join('Stocks','hpq.us.txt'),delimiter=',',usecols=['Date','Open','High','Low','Close'])
            print('Loaded data from the Kaggle repository')



