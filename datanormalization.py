from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
# import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler
import math

class Normalize:
    mid_prices = []
    window_len = 0
    train_data = []
    test_data = []

    def __init__(self, mid_prices, window_len):
        self.mid_prices = mid_prices
        self.window_len = window_len

    def normalize(self):
        train_len = math.floor(len(self.mid_prices) * 0.9)
        train_data = self.mid_prices[:train_len]
        test_data = self.mid_prices[train_len:]

        scaler = MinMaxScaler()
        train_data = train_data.reshape(-1,1)
        test_data = test_data.reshape(-1,1)

        smoothing_window_size = math.floor(train_len / self.window_len)
        for di in range(0,self.window_len * smoothing_window_size,smoothing_window_size):
            scaler.fit(train_data[di:di+smoothing_window_size,:])
            train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])

        if(train_len > self.window_len * smoothing_window_size):
            scaler.fit(train_data[di+smoothing_window_size:,:])
            train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:])

        train_data = train_data.reshape(-1)

        test_data = scaler.transform(test_data).reshape(-1)

        EMA = 0.0
        gamma = 0.1
        for ti in range(train_len):
            EMA = gamma*train_data[ti] + (1-gamma)*EMA
        train_data[ti] = EMA

        self.train_data = train_data
        self.test_data = test_data
        all_mid_data = np.concatenate([train_data,test_data],axis=0)

        return all_mid_data
    
    def get_train(self):
        return self.train_data
    
    def get_test(self):
        return self.test_data