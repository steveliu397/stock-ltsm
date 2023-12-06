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

dg = DataGeneratorSeq(train_data,5,5)
u_data, u_labels = dg.unroll_batches()


D = 1 # Dimensionality of the data. Since your data is 1-D this would be 1
num_unrollings = 50 # Number of time steps you look into the future.
batch_size = 500 # Number of samples in a batch
num_nodes = [200,200,150] # Number of hidden nodes in each layer of the deep LSTM stack we're using
n_layers = len(num_nodes) # number of layers
dropout = 0.2 # dropout amount

tf.compat.v1.reset_default_graph() # This is important in case you run this multiple times

train_inputs, train_outputs = [],[]

for ui in range(num_unrollings):
            tf.compat.v1.disable_eager_execution()
            train_inputs.append(tf.compat.v1.placeholder(tf.float32, shape=[batch_size,D],name='train_inputs_%d'%ui))
            train_outputs.append(tf.compat.v1.placeholder(tf.float32, shape=[batch_size,1], name = 'train_outputs_%d'%ui))

'''
optimized_model = LSTMOptimization(D, num_unrollings, batch_size, num_nodes, n_layers, dropout, train_inputs, train_outputs)
optimized_model.optimize()


inc_gstep = optimized_model.get_inc_gstep()
optimizer = optimized_model.get_optimizer()
loss = optimized_model.get_loss()
tf_learning_rate = optimized_model.get_tf_learning_rate()
tf_min_learning_rate = optimized_model.get_tf_min_learning_rate()
sample_inputs = optimized_model.get_sample_inputs()
sample_prediction = optimized_model.get_sample_prediction()
reset_sample_states = optimized_model.get_reset_sample_states()
'''

# Running the program
epochs = 30
valid_summary = 1 # Interval you make test predictions

n_predict_once = 50 # Number of steps you continously predict for

train_seq_length = len(train_data) # Full length of the training data

train_mse_ot = [] # Accumulate Train losses
test_mse_ot = [] # Accumulate Test loss
predictions_over_time = [] # Accumulate predictions





# Used for decaying learning rate
loss_nondecrease_count = 0
loss_nondecrease_threshold = 2 # If the test error hasn't increased in this many steps, decrease learning rate

print('Initialized')
average_loss = 0

# Define data generator
data_gen = DataGeneratorSeq(train_data,batch_size,num_unrollings)

x_axis_seq = []

# Points you start your test predictions from
# Changed input here for testing ---------------------------- need to integrate start, end and step into a method
test_points_seq = np.arange(5432,6032,50).tolist()



lstm_run = LSTMRun(D, num_unrollings, batch_size, num_nodes, n_layers, dropout, train_inputs, train_outputs,
                epochs, valid_summary, n_predict_once, train_seq_length, train_mse_ot, test_mse_ot, predictions_over_time, loss_nondecrease_count, loss_nondecrease_threshold, data_gen, average_loss, x_axis_seq, test_points_seq,
                all_mid_data)
lstm_run.run()

