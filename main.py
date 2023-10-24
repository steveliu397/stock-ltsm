from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
# import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler
from datareader import DataReader
from datagenerator import DataGeneratorSeq
from datanormalization import Normalize
import math
# from datareader import [class name]

# outsourced data builidng to datareader.py
ticker = 'SPY'
file_to_save = 'stock_market_data-%s.csv'%ticker

data_reader = DataReader(ticker, file_to_save)
data_reader.get_data()

df = pd.read_csv(file_to_save)

df.sort_values('Date')

high_prices = df.loc[:,'High'].values
low_prices = df.loc[:,'Low'].values
mid_prices = (high_prices+low_prices)/2.0

# outsourced data wrangling to datanormalization.py
window_len = 4
data_wrangler = Normalize(mid_prices, window_len)
all_mid_data = data_wrangler.normalize()
train_data = data_wrangler.get_train()
test_data = data_wrangler.get_test()

dg = DataGeneratorSeq(train_data,5,5)
u_data, u_labels = dg.unroll_batches()

for ui,(dat,lbl) in enumerate(zip(u_data,u_labels)):   
    print('\n\nUnrolled index %d'%ui)
    dat_ind = dat
    lbl_ind = lbl
    print('\tInputs: ',dat )
    print('\n\tOutput:',lbl)