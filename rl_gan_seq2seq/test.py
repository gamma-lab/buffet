from __future__ import print_function, division
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

num_epochs = 100
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_series_length//batch_size//truncated_backprop_length

def generateData():
    x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0

    x = x.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
    y = y.reshape((batch_size, -1))

    return (x, y)

batchX_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

cell_state = tf.placeholder(tf.float32, [batch_size, state_size])
hidden_state = tf.placeholder(tf.float32, [batch_size, state_size])
init_state = tf.nn.rnn_cell.LSTMStateTuple(cell_state, hidden_state)

W2 = tf.Variable(tf.random_normal(shape=[state_size,num_classes], mean=0, stddev=0.1),dtype=tf.float32)
b2 = tf.Variable(tf.random_normal(shape=[1,num_classes], mean=0, stddev=0.1),dtype=tf.float32)

# Unpack columns
inputs_series = tf.split(batchX_placeholder, truncated_backprop_length, axis=1)
#inputs_series = tf.split(1, truncated_backprop_length, batchX_placeholder)
labels_series = tf.unstack(batchY_placeholder, axis=1)

# Forward passes

def create_rnn_cell():
                # encoDecoCell = tf.contrib.rnn.GRUCell(  # Or GRUCell, LSTMCell(args.hiddenSize)
    encoDecoCell = tf.nn.rnn_cell.GRUCell(  # Or GRUCell, LSTMCell(args.hiddenSize)
                    state_size,
                )
    encoDecoCell = tf.contrib.rnn.DropoutWrapper(
                    encoDecoCell,
                    input_keep_prob=1.0,
                    output_keep_prob=0.8
                )
    return encoDecoCell
context_multi = tf.contrib.rnn.MultiRNNCell(
                [create_rnn_cell() for _ in range(1)],
            )
context_multi = tf.contrib.rnn.EmbeddingWrapper(context_multi, embedding_classes=15,
                                                                embedding_size=20)
states_series, current_state = tf.contrib.rnn.static_rnn(context_multi, inputs_series, dtype=tf.float32)
logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series,labels_series)]
total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)

run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

with tf.Session() as sess:
    loss_list = []
    sess.run(tf.initialize_all_variables())
    for epoch_idx in range(num_epochs):
        x,y = generateData()
        _current_cell_state = np.zeros((batch_size, state_size))
        _current_hidden_state = np.zeros((batch_size, state_size))

        print("New data, epoch", epoch_idx)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length

            batchX = x[:,start_idx:end_idx]
            batchY = y[:,start_idx:end_idx]

            _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [total_loss, train_step, current_state, predictions_series],
                feed_dict={
                    batchX_placeholder: batchX,
                    batchY_placeholder: batchY,
                },
                options=run_options,
                run_metadata=run_metadata)
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
        with open('timeline.json', 'w') as f:
            f.write(ctf)
            loss_list.append(_total_loss)

            if batch_idx%100 == 0:
                print("Step",batch_idx, "Batch loss", _total_loss)

