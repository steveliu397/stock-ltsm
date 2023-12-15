from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import math


class LSTMModel:

    def __init__(self, num_unrollings, batch_size, num_nodes, n_layers, dropout, train_inputs):
        lstm_cells = [
        tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=num_nodes[li],
                            state_is_tuple=True,
                            initializer= tf.keras.initializers.GlorotUniform()
                           )
        for li in range(n_layers)]

        drop_lstm_cells = [tf.compat.v1.nn.rnn_cell.DropoutWrapper(
            lstm, input_keep_prob=1.0,output_keep_prob=1.0-dropout, state_keep_prob=1.0-dropout
        ) for lstm in lstm_cells]
        drop_multi_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(drop_lstm_cells)
        self.multi_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(lstm_cells)

        self.w = tf.compat.v1.get_variable('w',shape=[num_nodes[-1], 1], initializer=tf.keras.initializers.GlorotUniform())
        self.b = tf.compat.v1.get_variable('b',initializer=tf.random.uniform([1],-0.1,0.1))
 
        self.c, self.h = [],[]
        initial_state = []
        for li in range(n_layers):
            self.c.append(tf.Variable(tf.zeros([batch_size, num_nodes[li]]), trainable=False))
            self.h.append(tf.Variable(tf.zeros([batch_size, num_nodes[li]]), trainable=False))
            initial_state.append(tf.compat.v1.nn.rnn_cell.LSTMStateTuple(self.c[li], self.h[li]))

        all_inputs = tf.concat([tf.expand_dims(t,0) for t in train_inputs],axis=0)

        all_lstm_outputs, self.state = tf.compat.v1.nn.dynamic_rnn(
            drop_multi_cell, all_inputs, initial_state=tuple(initial_state),
            time_major = True, dtype=tf.float32)

        all_lstm_outputs = tf.reshape(all_lstm_outputs, [batch_size*num_unrollings,num_nodes[-1]])

        all_outputs = tf.compat.v1.nn.xw_plus_b(all_lstm_outputs,self.w,self.b)

        self.split_outputs = tf.split(all_outputs,num_unrollings,axis=0)

    
    def get_c(self):
        return self.c
    
    def get_h(self):
        return self.h
    
    def get_w(self):
        return self.w
    
    def get_b(self):
        return self.b
    
    def get_multi_cell(self):
        return self.multi_cell
    
    def get_state(self):
        return self.state
    
    def get_split_outputs(self):
        return self.split_outputs
