
# You unroll the input over time defining placeholders for each time step
for ui in range(num_unrollings):
    tf.compat.v1.disable_eager_execution()
    train_inputs.append(tf.compat.v1.placeholder(tf.float32, shape=[batch_size,D],name='train_inputs_%d'%ui))
    train_outputs.append(tf.compat.v1.placeholder(tf.float32, shape=[batch_size,1], name = 'train_outputs_%d'%ui))


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
multi_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(lstm_cells)

w = tf.compat.v1.get_variable('w',shape=[num_nodes[-1], 1], initializer=tf.keras.initializers.GlorotUniform())
b = tf.compat.v1.get_variable('b',initializer=tf.random.uniform([1],-0.1,0.1))


# Create cell state and hidden state variables to maintain the state of the LSTM
c, h = [],[]
initial_state = []
for li in range(n_layers):
    c.append(tf.Variable(tf.zeros([batch_size, num_nodes[li]]), trainable=False))
    h.append(tf.Variable(tf.zeros([batch_size, num_nodes[li]]), trainable=False))
    initial_state.append(tf.compat.v1.nn.rnn_cell.LSTMStateTuple(c[li], h[li]))

# Do several tensor transofmations, because the function dynamic_rnn requires the output to be of
# a specific format. Read more at: https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn
all_inputs = tf.concat([tf.expand_dims(t,0) for t in train_inputs],axis=0)

# all_outputs is [seq_length, batch_size, num_nodes]
all_lstm_outputs, state = tf.compat.v1.nn.dynamic_rnn(
    drop_multi_cell, all_inputs, initial_state=tuple(initial_state),
    time_major = True, dtype=tf.float32)

all_lstm_outputs = tf.reshape(all_lstm_outputs, [batch_size*num_unrollings,num_nodes[-1]])

all_outputs = tf.compat.v1.nn.xw_plus_b(all_lstm_outputs,w,b)

split_outputs = tf.split(all_outputs,num_unrollings,axis=0)

# When calculating the loss you need to be careful about the exact form, because you calculate
# loss of all the unrolled steps at the same time
# Therefore, take the mean error or each batch and get the sum of that over all the unrolled steps

print('Defining training Loss')
loss = 0.0
with tf.control_dependencies([tf.compat.v1.assign(c[li], state[li][0]) for li in range(n_layers)]+
                             [tf.compat.v1.assign(h[li], state[li][1]) for li in range(n_layers)]):
    for ui in range(num_unrollings):
        loss += tf.reduce_mean(0.5*(split_outputs[ui]-train_outputs[ui])**2)

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

sample_inputs = tf.compat.v1.placeholder(tf.float32, shape=[1,D])

# Maintaining LSTM state for prediction stage
sample_c, sample_h, initial_sample_state = [],[],[]
for li in range(n_layers):
    sample_c.append(tf.Variable(tf.zeros([1, num_nodes[li]]), trainable=False))
    sample_h.append(tf.Variable(tf.zeros([1, num_nodes[li]]), trainable=False))
    initial_sample_state.append(tf.compat.v1.nn.rnn_cell.LSTMStateTuple(sample_c[li],sample_h[li]))

reset_sample_states = tf.group(*[tf.compat.v1.assign(sample_c[li],tf.zeros([1, num_nodes[li]])) for li in range(n_layers)],
                               *[tf.compat.v1.assign(sample_h[li],tf.zeros([1, num_nodes[li]])) for li in range(n_layers)])

sample_outputs, sample_state = tf.compat.v1.nn.dynamic_rnn(multi_cell, tf.expand_dims(sample_inputs,0),
                                   initial_state=tuple(initial_sample_state),
                                   time_major = True,
                                   dtype=tf.float32)

with tf.control_dependencies([tf.compat.v1.assign(sample_c[li],sample_state[li][0]) for li in range(n_layers)]+
                              [tf.compat.v1.assign(sample_h[li],sample_state[li][1]) for li in range(n_layers)]):  
  sample_prediction = tf.compat.v1.nn.xw_plus_b(tf.reshape(sample_outputs,[1,-1]), w, b)

print('\tAll done')



class LSTMOptimization:
    
    def __init__(self, D, num_unrollings, batch_size, num_nodes, n_layers, dropout, train_inputs, train_outputs):
        model = LSTMModel(D, num_unrollings, batch_size, num_nodes, n_layers, dropout, train_inputs)

        self.num_unrollings = num_unrollings
        self.num_nodes = num_nodes
        self.n_layers = n_layers
        self.train_outputs = train_outputs

        self.c = model.get_c()
        self.h = model.get_h()
        self.w = model.get_w()
        self.b = model.get_b()
        self.multi_cell = model.get_multi_cell()
        self.state = model.get_state()
        self.split_outputs = model.get_split_outputs()


    def optimize(self):
        print('Defining training Loss')
        self.loss = 0.0
        with tf.control_dependencies([tf.compat.v1.assign(self.c[li], self.state[li][0]) for li in range(self.n_layers)]+
                                    [tf.compat.v1.assign(self.h[li], self.state[li][1]) for li in range(self.n_layers)]):
            for ui in range(self.num_unrollings):
                self.loss += tf.reduce_mean(0.5*(self.split_outputs[ui]-self.train_outputs[ui])**2)

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

        self.sample_inputs = tf.compat.v1.placeholder(tf.float32, shape=[1,self.D])

        # Maintaining LSTM state for prediction stage
        sample_c, sample_h, initial_sample_state = [],[],[]
        for li in range(self.n_layers):
            sample_c.append(tf.Variable(tf.zeros([1, self.num_nodes[li]]), trainable=False))
            sample_h.append(tf.Variable(tf.zeros([1, self.num_nodes[li]]), trainable=False))
            initial_sample_state.append(tf.compat.v1.nn.rnn_cell.LSTMStateTuple(sample_c[li],sample_h[li]))

        self.reset_sample_states = tf.group(*[tf.compat.v1.assign(sample_c[li],tf.zeros([1, self.num_nodes[li]])) for li in range(self.n_layers)],
                                    *[tf.compat.v1.assign(sample_h[li],tf.zeros([1, self.num_nodes[li]])) for li in range(self.n_layers)])

        sample_outputs, sample_state = tf.compat.v1.nn.dynamic_rnn(self.multi_cell, tf.expand_dims(self.sample_inputs,0),
                                        initial_state=tuple(initial_sample_state),
                                        time_major = True,
                                        dtype=tf.float32)

        with tf.control_dependencies([tf.compat.v1.assign(sample_c[li],sample_state[li][0]) for li in range(self.n_layers)]+
                                    [tf.compat.v1.assign(sample_h[li],sample_state[li][1]) for li in range(self.n_layers)]):  
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


for ep in range(epochs):       

    # ========================= Training =====================================
    for step in range(train_seq_length//batch_size):

        u_data, u_labels = data_gen.unroll_batches()

        feed_dict = {}
        for ui,(dat,lbl) in enumerate(zip(u_data,u_labels)):            
            feed_dict[train_inputs[ui]] = dat.reshape(-1,1)
            feed_dict[train_outputs[ui]] = lbl.reshape(-1,1)

        feed_dict.update({tf_learning_rate: 0.0001, tf_min_learning_rate:0.000001})

        _, l = session.run([optimizer, loss], feed_dict=feed_dict)

        average_loss += l

    # ============================ Validation ==============================
    if (ep+1) % valid_summary == 0:

      average_loss = average_loss/(valid_summary*(train_seq_length//batch_size))

      # The average loss
      if (ep+1)%valid_summary==0:
        print('Average loss at step %d: %f' % (ep+1, average_loss))

      train_mse_ot.append(average_loss)

      average_loss = 0 # reset loss

      predictions_seq = []

      mse_test_loss_seq = []

      # ===================== Updating State and Making Predicitons ========================
      for w_i in test_points_seq:
        mse_test_loss = 0.0
        our_predictions = []

        if (ep+1)-valid_summary==0:
          # Only calculate x_axis values in the first validation epoch
          x_axis=[]

        # Feed in the recent past behavior of stock prices
        # to make predictions from that point onwards
        for tr_i in range(w_i-num_unrollings+1,w_i-1):
          current_price = all_mid_data[tr_i]
          feed_dict[sample_inputs] = np.array(current_price).reshape(1,1)    
          _ = session.run(sample_prediction,feed_dict=feed_dict)

        feed_dict = {}

        current_price = all_mid_data[w_i-1]

        feed_dict[sample_inputs] = np.array(current_price).reshape(1,1)

        # Make predictions for this many steps
        # Each prediction uses previous prediciton as it's current input
        for pred_i in range(n_predict_once):

          pred = session.run(sample_prediction,feed_dict=feed_dict)

          our_predictions.append(np.ndarray.item(pred))

          feed_dict[sample_inputs] = np.asarray(pred).reshape(-1,1)

          if (ep+1)-valid_summary==0:
            # Only calculate x_axis values in the first validation epoch
            x_axis.append(w_i+pred_i)

          mse_test_loss += 0.5*(pred-all_mid_data[w_i+pred_i])**2

        session.run(reset_sample_states)

        predictions_seq.append(np.array(our_predictions))

        mse_test_loss /= n_predict_once
        mse_test_loss_seq.append(mse_test_loss)

        if (ep+1)-valid_summary==0:
          x_axis_seq.append(x_axis)

      current_test_mse = np.mean(mse_test_loss_seq)

      # Learning rate decay logic
      if len(test_mse_ot)>0 and current_test_mse > min(test_mse_ot):
          loss_nondecrease_count += 1
      else:
          loss_nondecrease_count = 0

      if loss_nondecrease_count > loss_nondecrease_threshold :
            session.run(inc_gstep)
            loss_nondecrease_count = 0
            print('\tDecreasing learning rate by 0.5')

      test_mse_ot.append(current_test_mse)
      print('\tTest MSE: %.5f'%np.mean(mse_test_loss_seq))
      predictions_over_time.append(predictions_seq)
      print('\tFinished Predictions')



