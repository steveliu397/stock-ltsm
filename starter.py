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
# from datareader import [class name]

# outsourced data-builidng to datareader.py

ticker = 'AAL'

my_datareader = DataReader(ticker)

df = my_datareader.sort_values('Date')

df.head()

high_prices = df.loc[:,'High'].as_matrix()
low_prices = df.loc[:,'Low'].as_matrix()
mid_prices = (high_prices+low_prices)/2.0

train_data = mid_prices[:len(mid_prices) * 0.9]
test_data = mid_prices[len(mid_prices) * 0.1:]

print(len(mid_prices))