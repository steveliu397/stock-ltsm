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
from lstm_model import LSTMModel


class LSTMOptimization:
    
    def __init__(self, D, num_unrollings, batch_size, num_nodes, n_layers, dropout, train_inputs, train_outputs):
        model = LSTMModel(num_unrollings, batch_size, num_nodes, n_layers, dropout, train_inputs)

        self.c = model.get_c()
        self.h = model.get_h()
        self.w = model.get_w()
        self.b = model.get_b()
        self.multi_cell = model.get_multi_cell()
        self.state = model.get_state()
        self.split_outputs = model.get_split_outputs()


        print('Defining training Loss')
        self.loss = 0.0
        with tf.control_dependencies([tf.compat.v1.assign(self.c[li], self.state[li][0]) for li in range(n_layers)]+
                                    [tf.compat.v1.assign(self.h[li], self.state[li][1]) for li in range(n_layers)]):
            for ui in range(num_unrollings):
                self.loss += tf.reduce_mean(0.5*(self.split_outputs[ui]-train_outputs[ui])**2)

        print('Learning rate decay operations')
        global_step = tf.Variable(0, trainable=False)
        self.inc_gstep = tf.compat.v1.assign(global_step,global_step + 1)
        self.tf_learning_rate = tf.compat.v1.placeholder(shape=None,dtype=tf.float32)
        self.tf_min_learning_rate = tf.compat.v1.placeholder(shape=None,dtype=tf.float32)

        learning_rate = tf.maximum(
            tf.compat.v1.train.exponential_decay(self.tf_learning_rate, global_step, decay_steps=1, decay_rate=0.5, staircase=True),
            self.tf_min_learning_rate)

        # Optimizer.
        print('TF Optimization operations')
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
        gradients, v = zip(*self.optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        self.optimizer = self.optimizer.apply_gradients(
            zip(gradients, v))

        print('\tAll done')


        print('Defining prediction related TF functions')

        self.sample_inputs = tf.compat.v1.placeholder(tf.float32, shape=[1,D])

        # Maintaining LSTM state for prediction stage
        sample_c, sample_h, initial_sample_state = [],[],[]
        for li in range(n_layers):
            sample_c.append(tf.Variable(tf.zeros([1, num_nodes[li]]), trainable=False))
            sample_h.append(tf.Variable(tf.zeros([1, num_nodes[li]]), trainable=False))
            initial_sample_state.append(tf.compat.v1.nn.rnn_cell.LSTMStateTuple(sample_c[li],sample_h[li]))

        self.reset_sample_states = tf.group(*[tf.compat.v1.assign(sample_c[li],tf.zeros([1, num_nodes[li]])) for li in range(n_layers)],
                                    *[tf.compat.v1.assign(sample_h[li],tf.zeros([1, num_nodes[li]])) for li in range(n_layers)])

        sample_outputs, sample_state = tf.compat.v1.nn.dynamic_rnn(self.multi_cell, tf.expand_dims(self.sample_inputs,0),
                                        initial_state=tuple(initial_sample_state),
                                        time_major = True,
                                        dtype=tf.float32)

        with tf.control_dependencies([tf.compat.v1.assign(sample_c[li],sample_state[li][0]) for li in range(n_layers)]+
                                    [tf.compat.v1.assign(sample_h[li],sample_state[li][1]) for li in range(n_layers)]):  
            self.sample_prediction = tf.compat.v1.nn.xw_plus_b(tf.reshape(sample_outputs,[1,-1]), self.w, self.b)

        print('\tAll done')


    
    def get_inc_gstep(self):
        return self.inc_gstep

    def get_optimizer(self):
        return self.optimizer
    
    def get_loss(self):
        return self.loss
    
    def get_tf_learning_rate(self):
        return self.tf_learning_rate
    
    def get_tf_min_learning_rate(self):
        return self.tf_min_learning_rate
    
    def get_sample_inputs(self):
        return self.sample_inputs
    
    def get_sample_prediction(self):
        return self.sample_prediction
    
    def get_reset_sample_states(self):
        return self.reset_sample_states