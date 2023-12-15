from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from sklearn.preprocessing import MinMaxScaler
import math
from lstm_optimization import LSTMOptimization


# Outsourced code to run the LSTM, for consideration
class LSTMRun:

    def __init__(self, num_unrollings, batch_size, num_nodes, n_layers, dropout,
                 epochs, valid_summary, n_predict_once, train_seq_length,
                 loss_nondecrease_count, loss_nondecrease_threshold, data_gen, average_loss, test_points_seq,
                 all_mid_data):
        D = 1   # The data is 1D in this model

        train_inputs, train_outputs = [],[]

        for ui in range(num_unrollings):
                    tf.compat.v1.disable_eager_execution()
                    train_inputs.append(tf.compat.v1.placeholder(tf.float32, shape=[batch_size,D],name='train_inputs_%d'%ui))
                    train_outputs.append(tf.compat.v1.placeholder(tf.float32, shape=[batch_size,1], name = 'train_outputs_%d'%ui))

        optimized_model = LSTMOptimization(D, num_unrollings, batch_size, num_nodes, n_layers, dropout, train_inputs, train_outputs)

        self.num_unrollings = num_unrollings
        self.train_inputs = train_inputs
        self.train_outputs = train_outputs
        self.batch_size = batch_size

        self.epochs = epochs
        self.valid_summary = valid_summary
        self.n_predict_once = n_predict_once
        self.train_seq_length = train_seq_length
        self.train_mse_ot = []
        self.test_mse_ot = []
        self.predictions_over_time = []
        
        self.loss_nondecrease_count = loss_nondecrease_count
        self.loss_nondecrease_threshold = loss_nondecrease_threshold
        self.data_gen = data_gen
        self.average_loss = average_loss
        self.x_axis_seq = []
        self.test_points_seq = test_points_seq

        self.all_mid_data = all_mid_data

        self.inc_gstep = optimized_model.get_inc_gstep()
        self.optimizer = optimized_model.get_optimizer()
        self.loss = optimized_model.get_loss()
        self.tf_learning_rate = optimized_model.get_tf_learning_rate()
        self.tf_min_learning_rate = optimized_model.get_tf_min_learning_rate()
        self.sample_inputs = optimized_model.get_sample_inputs()
        self.sample_prediction = optimized_model.get_sample_prediction()
        self.reset_sample_states = optimized_model.get_reset_sample_states()


    def run(self):
        session = tf.compat.v1.InteractiveSession()
        init = tf.compat.v1.global_variables_initializer()
        session.run(init)

        for ep in range(self.epochs):       

            # ========================= Training =====================================
            for step in range(self.train_seq_length//self.batch_size):

                u_data, u_labels = self.data_gen.unroll_batches()

                feed_dict = {}
                for ui,(dat,lbl) in enumerate(zip(u_data,u_labels)):            
                    feed_dict[self.train_inputs[ui]] = dat.reshape(-1,1)
                    feed_dict[self.train_outputs[ui]] = lbl.reshape(-1,1)

                feed_dict.update({self.tf_learning_rate: 0.0001, self.tf_min_learning_rate:0.000001})

                _, l = session.run([self.optimizer, self.loss], feed_dict=feed_dict)

                self.average_loss += l

            # ============================ Validation ==============================
            if (ep+1) % self.valid_summary == 0:

                self.average_loss = self.average_loss/(self.valid_summary*(self.train_seq_length//self.batch_size))

                # The average loss
                if (ep+1)%self.valid_summary==0:
                    print('Average loss at step %d: %f' % (ep+1, self.average_loss))

                self.train_mse_ot.append(self.average_loss)

                self.average_loss = 0 # reset loss

                predictions_seq = []

                mse_test_loss_seq = []

                # ===================== Updating State and Making Predicitons ========================
                for w_i in self.test_points_seq:
                    mse_test_loss = 0.0
                    our_predictions = []

                    if (ep+1)-self.valid_summary==0:
                        # Only calculate x_axis values in the first validation epoch
                        x_axis=[]

                    # Feed in the recent past behavior of stock prices
                    # to make predictions from that point onwards
                    for tr_i in range(w_i-self.num_unrollings+1,w_i-1):
                        current_price = self.all_mid_data[tr_i]
                        feed_dict[self.sample_inputs] = np.array(current_price).reshape(1,1)    
                        _ = session.run(self.sample_prediction,feed_dict=feed_dict)

                    feed_dict = {}

                    current_price = self.all_mid_data[w_i-1]

                    feed_dict[self.sample_inputs] = np.array(current_price).reshape(1,1)

                    # Make predictions for this many steps
                    # Each prediction uses previous prediciton as it's current input
                    for pred_i in range(self.n_predict_once):

                        pred = session.run(self.sample_prediction,feed_dict=feed_dict)

                        our_predictions.append(np.ndarray.item(pred))

                        feed_dict[self.sample_inputs] = np.asarray(pred).reshape(-1,1)

                        if (ep+1)-self.valid_summary==0:
                            # Only calculate x_axis values in the first validation epoch
                            x_axis.append(w_i+pred_i)

                        mse_test_loss += 0.5*(pred-self.all_mid_data[w_i+pred_i])**2

                    session.run(self.reset_sample_states)

                    predictions_seq.append(np.array(our_predictions))

                    mse_test_loss /= self.n_predict_once
                    mse_test_loss_seq.append(mse_test_loss)

                    if (ep+1)-self.valid_summary==0:
                        self.x_axis_seq.append(x_axis)

                current_test_mse = np.mean(mse_test_loss_seq)

                # Learning rate decay logic
                if len(self.test_mse_ot)>0 and current_test_mse > min(self.test_mse_ot):
                    self.loss_nondecrease_count += 1
                else:
                    self.loss_nondecrease_count = 0

                if self.loss_nondecrease_count > self.loss_nondecrease_threshold :
                    session.run(self.inc_gstep)
                    self.loss_nondecrease_count = 0
                    print('\tDecreasing learning rate by 0.5')

                self.test_mse_ot.append(current_test_mse)
                print('\tTest MSE: %.5f'%np.mean(mse_test_loss_seq))
                self.predictions_over_time.append(predictions_seq)
                print('\tFinished Predictions')