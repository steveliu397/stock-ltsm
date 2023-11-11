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


class Definition:
    D = 0
    num_unrollings = 0
    batch_size = 0
    num_nodes = []
    n_layers = len(num_nodes)
    dropout = 0.0
    train_inputs = []
    train_outputs = []

    drop_multi_cell = []
    multi_cell = []
    w = [[]]
    b = []
    state = [[]]
    split_outputs = []
    c = []
    h = []

    def __init__(self, D, num_unrollings, batch_size, num_nodes, n_layers, dropout, train_inputs, train_outputs):
        self.D = D
        self.num_unrollings = num_unrollings
        self.batch_size = batch_size
        self.num_nodes = num_nodes
        self.n_layers = n_layers
        self.dropout = dropout

        for ui in range(num_unrollings):
            tf.compat.v1.disable_eager_execution()
            train_inputs.append(tf.compat.v1.placeholder(tf.float32, shape=[batch_size,D],name='train_inputs_%d'%ui))
            train_outputs.append(tf.compat.v1.placeholder(tf.float32, shape=[batch_size,1], name = 'train_outputs_%d'%ui))

        self.train_inputs = train_inputs
        self.train_outputs = train_outputs

    
    def define_layers(self):
        lstm_cells = [
        tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=self.num_nodes[li],
                            state_is_tuple=True,
                            initializer= tf.keras.initializers.GlorotUniform()
                           )
        for li in range(self.n_layers)]

        drop_lstm_cells = [tf.compat.v1.nn.rnn_cell.DropoutWrapper(
            lstm, input_keep_prob=1.0,output_keep_prob=1.0-self.dropout, state_keep_prob=1.0-self.dropout
        ) for lstm in lstm_cells]
        self.drop_multi_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(drop_lstm_cells)
        self.multi_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(lstm_cells)

        self.w = tf.compat.v1.get_variable('w',shape=[self.num_nodes[-1], 1], initializer=tf.keras.initializers.GlorotUniform())
        self.b = tf.compat.v1.get_variable('b',initializer=tf.random.uniform([1],-0.1,0.1))

    
    def regression(self):
        c, h = [],[]
        initial_state = []
        for li in range(self.n_layers):
            c.append(tf.Variable(tf.zeros([self.batch_size, self.num_nodes[li]]), trainable=False))
            h.append(tf.Variable(tf.zeros([self.batch_size, self.num_nodes[li]]), trainable=False))
            initial_state.append(tf.compat.v1.nn.rnn_cell.LSTMStateTuple(c[li], h[li]))

        all_inputs = tf.concat([tf.expand_dims(t,0) for t in self.train_inputs],axis=0)

        all_lstm_outputs, self.state = tf.compat.v1.nn.dynamic_rnn(
            self.drop_multi_cell, all_inputs, initial_state=tuple(initial_state),
            time_major = True, dtype=tf.float32)

        all_lstm_outputs = tf.reshape(all_lstm_outputs, [self.batch_size*self.num_unrollings,self.num_nodes[-1]])

        all_outputs = tf.compat.v1.nn.xw_plus_b(all_lstm_outputs,self.w,self.b)

        self.split_outputs = tf.split(all_outputs,self.num_unrollings,axis=0)

        self.c = c
        self.h = h


    def calculate(self):
        print('Defining training Loss')
        loss = 0.0
        with tf.control_dependencies([tf.compat.v1.assign(self.c[li], self.state[li][0]) for li in range(self.n_layers)]+
                                    [tf.compat.v1.assign(self.h[li], self.state[li][1]) for li in range(self.n_layers)]):
            for ui in range(self.num_unrollings):
                loss += tf.reduce_mean(0.5*(self.split_outputs[ui]-self.train_outputs[ui])**2)

        print('Learning rate decay operations')
        global_step = tf.Variable(0, trainable=False)
        inc_gstep = tf.compat.v1.assign(global_step,global_step + 1)
        tf_learning_rate = tf.compat.v1.placeholder(shape=None,dtype=tf.float32)
        tf_min_learning_rate = tf.compat.v1.placeholder(shape=None,dtype=tf.float32)

        learning_rate = tf.maximum(
            tf.compat.v1.train.exponential_decay(tf_learning_rate, global_step, decay_steps=1, decay_rate=0.5, staircase=True),
            tf_min_learning_rate)

        # Optimizer.
        print('TF Optimization operations')
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
        gradients, v = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        optimizer = optimizer.apply_gradients(
            zip(gradients, v))

        print('\tAll done')


        print('Defining prediction related TF functions')

        sample_inputs = tf.compat.v1.placeholder(tf.float32, shape=[1,self.D])

        # Maintaining LSTM state for prediction stage
        sample_c, sample_h, initial_sample_state = [],[],[]
        for li in range(self.n_layers):
            sample_c.append(tf.Variable(tf.zeros([1, self.num_nodes[li]]), trainable=False))
            sample_h.append(tf.Variable(tf.zeros([1, self.num_nodes[li]]), trainable=False))
            initial_sample_state.append(tf.compat.v1.nn.rnn_cell.LSTMStateTuple(sample_c[li],sample_h[li]))

        reset_sample_states = tf.group(*[tf.compat.v1.assign(sample_c[li],tf.zeros([1, self.num_nodes[li]])) for li in range(self.n_layers)],
                                    *[tf.compat.v1.assign(sample_h[li],tf.zeros([1, self.num_nodes[li]])) for li in range(self.n_layers)])

        sample_outputs, sample_state = tf.compat.v1.nn.dynamic_rnn(self.multi_cell, tf.expand_dims(sample_inputs,0),
                                        initial_state=tuple(initial_sample_state),
                                        time_major = True,
                                        dtype=tf.float32)

        with tf.control_dependencies([tf.compat.v1.assign(sample_c[li],sample_state[li][0]) for li in range(self.n_layers)]+
                                    [tf.compat.v1.assign(sample_h[li],sample_state[li][1]) for li in range(self.n_layers)]):  
            sample_prediction = tf.compat.v1.nn.xw_plus_b(tf.reshape(sample_outputs,[1,-1]), self.w, self.b)

        print('\tAll done')

        return inc_gstep, optimizer, loss, tf_learning_rate, tf_min_learning_rate, sample_inputs, sample_prediction, reset_sample_states