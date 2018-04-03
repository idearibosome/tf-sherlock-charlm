import os

import numpy as np
import tensorflow as tf


# configurations
NUM_STEPS = 100000

HIDE_GPUS = True

BATCH_SIZE = 32

NUM_RNN_UNITS = 128 # the number of RNN units
NUM_RNN_LAYERS = 1 # the number of RNN layers

SEQUENCE_LENGTH = 32 # the number of characters to be trained
SAMPLE_SEQUENCE_LENGTH = 256 # the number of characters to be sampled

LEARNING_RATE = 1e-3 # learning rate


def main(argv=None):
  
  # disable GPUs
  if (HIDE_GPUS):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  
  
  # load data
  with open('sherlock.txt', 'r') as f:
    data = f.read()
  data_size = len(data)
  print(' - data with %d characters is loaded' % (data_size))
  
  
  # construct vocabularies
  vocab = list(set(data))
  vocab_size = len(vocab)
  print(' - %d characters are unique' % (vocab_size))
  
  char_to_vocab = { ch:i for i, ch in enumerate(vocab) }
  vocab_to_char = { i:ch for i, ch in enumerate(vocab) }
  
  g = tf.Graph()
  with g.as_default():
    
    # placeholders
    input_data = tf.placeholder(tf.float32, [BATCH_SIZE, SEQUENCE_LENGTH, vocab_size])
    label_data = tf.placeholder(tf.int64, [BATCH_SIZE, SEQUENCE_LENGTH])
    process_sample_data = tf.placeholder(tf.bool, [])
    sample_initial_data = tf.placeholder(tf.float32, [vocab_size])
    
    def lstm_cell():
      return tf.contrib.rnn.BasicLSTMCell(
          num_units=NUM_RNN_UNITS,
          reuse=tf.get_variable_scope().reuse,
          state_is_tuple=True
      )
    
    with tf.variable_scope('rnn'):
      
      cell = tf.contrib.rnn.MultiRNNCell(
          cells=[lstm_cell() for _ in range(NUM_RNN_LAYERS)],
          state_is_tuple=True
      )
      
      output_w = tf.get_variable('output_w', [NUM_RNN_UNITS, vocab_size], dtype=tf.float32)
      output_b = tf.get_variable('output_b', [vocab_size], dtype=tf.float32)
      
      global_step = tf.train.get_or_create_global_step()
      
      state = cell.zero_state(BATCH_SIZE, tf.float32)
      last_state = state
      
      input_reshape = tf.reshape(input_data, [BATCH_SIZE, SEQUENCE_LENGTH, vocab_size])
      input_reshape = tf.transpose(input_reshape, [1, 0, 2])
      input_reshape = tf.unstack(input_reshape, num=SEQUENCE_LENGTH, axis=0)
      output, last_state = tf.nn.static_rnn(cell, input_reshape, initial_state=state)
      # output: sequence of [batch_size, num_rnn_units]
      output = tf.stack(output, axis=1)
      output = tf.reshape(output, [BATCH_SIZE*SEQUENCE_LENGTH, NUM_RNN_UNITS])
      
      # num_rnn_units -> vocab_size
      logit = tf.matmul(output, output_w) + output_b
      
      logit_reshape = tf.reshape(logit, [BATCH_SIZE, SEQUENCE_LENGTH, vocab_size])
      label_reshape = tf.reshape(label_data, [BATCH_SIZE, SEQUENCE_LENGTH])
      
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit_reshape, labels=label_reshape)
      total_loss = tf.reduce_mean(loss)
      
      optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
      train_op = optimizer.minimize(total_loss, global_step=global_step)
    
          
    # generate samples
    with tf.control_dependencies([train_op]):
      def generate_sample_data():
        with tf.variable_scope('rnn', reuse=True):
          def generate_sample_loop_fn(time, cell_output, cell_state, loop_state):
            finished = (time >= SAMPLE_SEQUENCE_LENGTH)
            
            if cell_output is None: # time == 0
              emit_output = None
              next_input = sample_initial_data
              
              next_cell_state = cell.zero_state(1, tf.float32)
            else:
              emit_output = cell_output
              
              output_logit = tf.matmul(cell_output, output_w) + output_b
              output_logit = tf.reshape(output_logit, [1, vocab_size])
              output_prob = tf.nn.softmax(output_logit, axis=-1)
              
              # draw next input based on softmax output
              output_class = tf.multinomial(tf.log(output_prob), 1)
              output_class = tf.reshape(output_class, [1])
              next_input = tf.one_hot(output_class, vocab_size, dtype=tf.float32)
              
              next_cell_state = cell_state
            
            next_input = tf.reshape(next_input, [1, vocab_size])
            next_loop_state = None
            
            return (finished, next_input, next_cell_state, emit_output, next_loop_state)
          
          sample_output_ta, _, _ = tf.nn.raw_rnn(cell, generate_sample_loop_fn)
          sample_output = sample_output_ta.stack()
          sample_output = tf.reshape(sample_output, [SAMPLE_SEQUENCE_LENGTH, NUM_RNN_UNITS])
          sample_output = tf.matmul(sample_output, output_w) + output_b
          sample_output = tf.nn.softmax(sample_output, axis=-1)
          
          # use argmax instead of drawing from tf.multinomial
          sample_output = tf.argmax(sample_output, axis=-1)
        
        return sample_output
      
      # generate sample only when process_sample_data=True
      sample_output = tf.cond(process_sample_data, generate_sample_data, lambda: tf.zeros([SAMPLE_SEQUENCE_LENGTH], dtype=tf.int64))
    
    # initializer
    init = tf.global_variables_initializer()
  
  
  # session
  sess = tf.Session(graph=g)
  sess.run(init)
  
  
  # run
  for step in range(NUM_STEPS):
    
    # train
    r_input_data_list = []
    r_label_data_list = []
    
    for _ in range(BATCH_SIZE):
      data_startindex = np.random.randint(data_size-SEQUENCE_LENGTH-1)
      
      r_input_data = []
      for i in range(SEQUENCE_LENGTH):
        current_data = [0.0] * vocab_size
        current_data[char_to_vocab[data[data_startindex+i]]] = 1.0
        r_input_data.append(current_data)
      
      r_label_data = [char_to_vocab[i] for i in data[(data_startindex+1):(data_startindex+SEQUENCE_LENGTH+1)]]
      
      r_input_data_list.append(r_input_data)
      r_label_data_list.append(r_label_data)
    
    r_sample_initial_data = [0.0] * vocab_size
    r_sample_initial_data[char_to_vocab[data[data_startindex]]] = 1.0
    
    feed_dict = {}
    feed_dict[input_data] = r_input_data_list
    feed_dict[label_data] = r_label_data_list
    feed_dict[sample_initial_data] = r_sample_initial_data
    feed_dict[process_sample_data] = (step % 100 == 0)
    
    (_, r_total_loss, r_sample_output) = sess.run(
        [train_op, total_loss, sample_output],
        feed_dict=feed_dict
    )    
    
    # sample
    if step % 100 == 0:
      print(' - step %d, loss %f' % (step, r_total_loss))
      
      r_sample_output_to_char = ''.join([vocab_to_char[i] for i in r_sample_output])
      print(r_sample_output_to_char)


if __name__ == '__main__':
  tf.app.run()