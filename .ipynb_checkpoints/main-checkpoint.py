from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from sklearn.preprocessing import MinMaxScaler
from datareader import DataReader
from datagenerator import DataGeneratorSeq
from datanormalization import Normalize
from lstm_model import LSTMModel
from lstm_optimization import LSTMOptimization
from lstm_run import LSTMRun
import math


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


num_unrollings = 50 # Number of time steps you look into the future.
batch_size = 500 # Number of samples in a batch
num_nodes = [200,200,150] # Number of hidden nodes in each layer of the deep LSTM stack we're using
n_layers = len(num_nodes) # number of layers
dropout = 0.2 # dropout amount

tf.compat.v1.reset_default_graph() # This is important in case you run this multiple times


# Running the program
epochs = 30
valid_summary = 1 # Interval you make test predictions
n_predict_once = 50 # Number of steps you continously predict for
train_seq_length = len(train_data) # Full length of the training data

x_axis_seq = []
predictions_over_time = []


# Used for decaying learning rate
loss_nondecrease_count = 0
loss_nondecrease_threshold = 2 # If the test error hasn't increased in this many steps, decrease learning rate

print('Initialized')
average_loss = 0

# Define data generator
data_gen = DataGeneratorSeq(train_data,batch_size,num_unrollings)


# Points you start your test predictions from
# Changed input here for testing ---------------------------- need to integrate start, end and step into a method
test_points_seq = np.arange(5432,6032,50).tolist()


lstm_run = LSTMRun(num_unrollings, batch_size, num_nodes, n_layers, dropout, epochs, valid_summary, n_predict_once, 
                   train_seq_length, loss_nondecrease_count, loss_nondecrease_threshold, data_gen, average_loss, 
                   test_points_seq, all_mid_data, x_axis_seq, predictions_over_time)
lstm_run.run()



best_prediction_epoch = 28 # replace this with the epoch that you got the best results when running the plotting code

plt.figure(figsize = (18,18))
plt.subplot(2,1,1)
plt.plot(range(df.shape[0]),all_mid_data,color='b')

# Plotting how the predictions change over time
# Plot older predictions with low alpha and newer predictions with high alpha
start_alpha = 0.25
alpha  = np.arange(start_alpha,1.1,(1.0-start_alpha)/len(predictions_over_time[::3]))
for p_i,p in enumerate(predictions_over_time[::3]):
    for xval,yval in zip(x_axis_seq,p):
        plt.plot(xval,yval,color='r',alpha=alpha[p_i])

plt.title('Evolution of Test Predictions Over Time',fontsize=18)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Mid Price',fontsize=18)
plt.xlim(11000,12500)

plt.subplot(2,1,2)

# Predicting the best test prediction you got
plt.plot(range(df.shape[0]),all_mid_data,color='b')
for xval,yval in zip(x_axis_seq,predictions_over_time[best_prediction_epoch]):
    plt.plot(xval,yval,color='r')

plt.title('Best Test Predictions Over Time',fontsize=18)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Mid Price',fontsize=18)
plt.xlim(11000,12500)
plt.show()