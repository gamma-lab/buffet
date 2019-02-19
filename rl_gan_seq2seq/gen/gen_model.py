from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import sys
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import utils.data_utils as data_utils
import gen.seq2seq as rl_seq2seq
from tensorflow.python.ops import variable_scope
from gen.tf_utils import EmbeddingWrapper as EmbeddingWrapper_GPU

sys.path.append('../utils')


class Seq2SeqModel(object):
    # Generator model
    def __init__(self, config, name_scope, forward_only=False, num_samples=256, dtype=tf.float32):

        # self.scope_name = scope_name
        # with tf.variable_scope(self.scope_name):
        with tf.device("/gpu:0"):
            source_vocab_size = config.vocab_size
            target_vocab_size = config.vocab_size
            emb_dim = config.emb_dim
            word_embedding_size = config.word_embedding
            dropout = config.keep_prob
            self.config = config
            self.buckets = config.buckets
            self.learning_rate = tf.Variable(float(config.learning_rate), trainable=False, dtype=dtype)
            self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * config.learning_rate_decay_factor)
            self.global_step = tf.Variable(0, trainable=False)
            self.batch_size = config.batch_size
            self.num_layers = config.num_layers
            self.max_gradient_norm = config.max_gradient_norm
            self.mc_search = tf.placeholder(tf.bool, name="mc_search")
            self.mc_position = tf.placeholder(tf.int32, name="mc_position")
            self.forward_only = tf.placeholder(tf.bool, name="forward_only")
            self.teacher_forcing = tf.placeholder(tf.bool, name="teacher_forcing")
            self.up_reward = tf.placeholder(tf.bool, name="up_reward")
            self.reward_bias = tf.get_variable("reward_bias", [1], dtype=tf.float32)
            self.ent_weight = float(config.ent_weight)
            # If we use sampled softmax, we need an output projection.
            output_projection = None
            softmax_loss_function = None
            # Sampled softmax only makes sense if we sample less than vocabulary size.
            # if num_samples > 0 and num_samples < target_vocab_size:
            if num_samples > 0 and num_samples < target_vocab_size:
                w_t = tf.get_variable("proj_w", [target_vocab_size, emb_dim], dtype=dtype)
                w = tf.transpose(w_t)
                b = tf.get_variable("proj_b", [target_vocab_size], dtype=dtype)
                output_projection = (w, b)

                def sampled_loss(inputs, labels):
                    labels = tf.reshape(labels, [-1, 1])
                    # We need to compute the sampled_softmax_loss using 32bit floats to
                    # avoid numerical instabilities.
                    local_w_t = tf.cast(w_t, tf.float32)
                    local_b = tf.cast(b, tf.float32)
                    local_inputs = tf.cast(inputs, tf.float32)
                    return tf.cast(
                        tf.nn.sampled_softmax_loss(local_w_t, local_b, labels, local_inputs,
                                                   num_samples, target_vocab_size), dtype)

                # softmax_loss_function = sampled_loss
                softmax_loss_function = None

            # Creation of the rnn cell
            def create_rnn_cell():
                encoDecoCell = tf.contrib.rnn.GRUCell(  # Or GRUCell, LSTMCell(args.hiddenSize)
                    emb_dim,
                )
                encoDecoCell = tf.contrib.rnn.DropoutWrapper(
                    encoDecoCell,
                    input_keep_prob=1.0,
                    output_keep_prob=dropout
                )
                return encoDecoCell

            single_cell = tf.contrib.rnn.MultiRNNCell(
                [create_rnn_cell() for _ in range(self.num_layers)],
            )
            # Create the internal multi-layer cell for our RNN.
            # single_cell = tf.contrib.rnn.GRUCell(emb_dim)
            # single_cell = tf.contrib.rnn.BasicLSTMCell(emb_dim)
            cell = single_cell

            # if self.num_layers > 1:
            #     cell = tf.contrib.rnn.MultiRNNCell([single_cell] * self.num_layers)

            # The seq2seq function: we use embedding for the input and attention.
            def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
                return rl_seq2seq.embedding_attention_seq2seq(
                    encoder_inputs,
                    decoder_inputs,
                    cell,
                    num_encoder_symbols=source_vocab_size,
                    num_decoder_symbols=target_vocab_size,
                    embedding_size=word_embedding_size,
                    output_projection=output_projection,
                    feed_previous=do_decode,
                    mc_search=self.mc_search,
                    dtype=dtype,
                    mc_position=self.mc_position)

            # Feeds for inputs.
            self.encoder_inputs = []
            self.decoder_inputs = []
            self.target_weights = []
            self.targets_input = []
            self.mc_sents = []
            for i in xrange(self.buckets[-1][0]):  # Last bucket is the biggest one.
                self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
            for i in xrange(self.buckets[-1][1] + 1):
                self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
                self.target_weights.append(tf.placeholder(dtype, shape=[None], name="weight{0}".format(i)))
                self.targets_input.append(tf.placeholder(tf.int32, shape=[None], name="target{0}".format(i)))

            # self.reward = [tf.placeholder(tf.float32, name="reward_%i" % i) for i in range(len(self.buckets))]
            self.reward = [tf.placeholder(tf.float32, shape=[None, None], name="reward_%i" % i) for i in
                           range(len(self.buckets))]

            # Our targets are decoder inputs shifted by one.
            # targets = [self.decoder_inputs[i + 1] for i in xrange(len(self.decoder_inputs) - 1)]

            self.outputs, self.losses, self.encoder_state, self.ent, self.mc_sents = rl_seq2seq.model_with_buckets(
                self.encoder_inputs,
                self.decoder_inputs,
                self.targets_input,
                self.target_weights,
                self.reward,
                self.buckets,
                source_vocab_size,
                self.batch_size,
                lambda x, y: seq2seq_f(x, y, tf.where(self.forward_only, True, False)),
                output_projection=output_projection,
                softmax_loss_function=softmax_loss_function)
            #
            for b in xrange(len(self.buckets)):
                self.outputs[b] = [
                    tf.cond(
                        self.forward_only,
                        lambda: tf.matmul(output, output_projection[0]) + output_projection[1],
                        lambda: output
                    )
                    for output in self.outputs[b]
                    ]

            #
            # forward_only==False  ----> adversarial learning
            if not forward_only:
                with tf.name_scope("gradient_descent"):
                    self.gradient_norms = []
                    self.updates = []
                    self.aj_losses = []
                    self.gen_params = [p for p in tf.trainable_variables() if name_scope in p.name]
                    # opt = tf.train.GradientDescentOptimizer(self.learning_rate)
                    # '''
                    opt = tf.train.AdamOptimizer(
                        learning_rate=self.learning_rate,
                        beta1=0.9,
                        beta2=0.999,
                        epsilon=1e-08
                    )
                    # '''
                    for b in xrange(len(self.buckets)):
                        # R =  tf.subtract(self.reward[b], self.reward_bias)
                        # self.reward[b] = self.reward[b] - reward_bias
                        # adjusted_loss = tf.cond(self.up_reward,
                        #                           lambda:tf.multiply(self.losses[b], self.reward[b]),
                        #                           lambda: self.losses[b])

                        adjusted_loss = self.losses[b]
                        adjusted_loss = tf.cond(
                            self.teacher_forcing,
                            lambda: adjusted_loss,
                            lambda: adjusted_loss + self.ent_weight * self.ent[b]
                        )
                        # adjusted_loss -= self.ent_weight * self.ent[b]
                        # if up_reward==true, lambda:tf.multiply(self.losses[b], self.reward[b]) will be executed

                        # adjusted_loss =  tf.cond(self.up_reward,
                        #                           lambda: tf.multiply(self.losses[b], R),
                        #                           lambda: self.losses[b])
                        self.aj_losses.append(adjusted_loss)
                        gradients, variables = zip(*opt.compute_gradients(adjusted_loss))
                        capped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                        optimizer = opt.apply_gradients(zip(capped_gradients, variables), global_step=self.global_step)
                        self.updates.append(optimizer)
                        # self.updates.append(opt.minimize(adjusted_loss, global_step=self.global_step))

            self.gen_variables = [k for k in tf.global_variables() if name_scope in k.name]
            self.saver = tf.train.Saver(self.gen_variables)

    def step(self, session, encoder_inputs, decoder_inputs, targets_input, target_weights,
             bucket_id, forward_only=True, reward=1, mc_search=False, teacher_forcing=False, up_reward=False,
             debug=True, mc_position=0):
        # Check if the sizes match.
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                             " %d != %d." % (len(target_weights), decoder_size))

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.

        input_feed = {
            self.forward_only.name: forward_only,
            self.teacher_forcing.name: teacher_forcing,
            self.up_reward.name: up_reward,
            self.mc_search.name: mc_search,
            self.mc_position.name: mc_position
        }

        for l in xrange(len(self.buckets)):
            input_feed[self.reward[l].name] = [[1.0] * len(encoder_inputs[0])] * self.buckets[l][1]
        if reward != 1:
            input_feed[self.reward[bucket_id].name] = reward

        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]
            input_feed[self.targets_input[l].name] = targets_input[l]

        # Since our targets are decoder inputs shifted by one, we need one more.
        # last_target = self.decoder_inputs[decoder_size].name
        # input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # Output feed: depends on whether we do a backward step or not.
        # forward_only==False  ----> adversarial learning
        if not forward_only:
            output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                           self.aj_losses[bucket_id],  # Gradient norm.
                           self.losses[bucket_id],
                           self.learning_rate_decay_op]  # Loss for this batch.
        # forward_only==True  ----> testing or sampling
        else:
            output_feed = [self.encoder_state[bucket_id],
                           # self.losses[bucket_id],
                           self.mc_sents[bucket_id]]  # Loss for this batch.
            for l in xrange(decoder_size):  # Output logits.
                output_feed.append(self.outputs[bucket_id][l])

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], [], outputs[0]  # Gradient norm, loss, no outputs.
        else:
            return outputs[0], [], outputs[1], outputs[2:]  # encoder_state, loss, outputs.

    def get_batch(self, train_data, bucket_id, batch_size, type=0):

        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs, targets_raw = [], [], []

        # pad them if needed, reverse encoder inputs and add GO to decoder.
        batch_source_encoder, batch_source_decoder = [], []
        # print("bucket_id: %s" %bucket_id)
        if type == 1:
            batch_size = 1
        for batch_i in xrange(batch_size):
            if type == 1:
                encoder_input, decoder_input = train_data[bucket_id]
            elif type == 2:
                # print("disc_data[bucket_id]: ", disc_data[bucket_id][0])
                encoder_input_a, decoder_input = train_data[bucket_id][0]
                encoder_input = encoder_input_a[batch_i]
            elif type == 0:
                encoder_input, decoder_input = random.choice(train_data[bucket_id])
            elif type == 3:
                encoder_input, decoder_input = train_data[batch_i]
                # print("train en: %s, de: %s" %(encoder_input, decoder_input))

            batch_source_encoder.append(encoder_input)
            batch_source_decoder.append(decoder_input)
            # Encoder inputs are padded and then reversed.
            encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
            if self.config.data_id == 'twitter':
                encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
            else:
                encoder_inputs.append(encoder_pad + encoder_input)
            # encoder_inputs.append(encoder_pad + encoder_input)

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            de_line = [data_utils.GO_ID] + decoder_input + [data_utils.EOS_ID] + [data_utils.PAD_ID] * (
            decoder_pad_size - 1)
            decoder_inputs.append(de_line[:decoder_size])
            ta_line = decoder_input + [data_utils.EOS_ID] + [data_utils.PAD_ID] * decoder_pad_size
            targets_raw.append(ta_line[:decoder_size])

        # Now we create batch-major vectors from the disc_data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights, batch_targets = [], [], [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(batch_size)], dtype=np.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(batch_size)], dtype=np.int32))

            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(batch_size, dtype=np.float32)
            for batch_idx in xrange(batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                if targets_raw[batch_idx][length_idx] == data_utils.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)

        # Batch targets inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(decoder_size):
            batch_targets.append(
                np.array([targets_raw[batch_idx][length_idx]
                          for batch_idx in xrange(batch_size)], dtype=np.int32))

        return (batch_encoder_inputs, batch_decoder_inputs, batch_weights, batch_source_encoder, batch_source_decoder,
                batch_targets)

    def gen_batch_preprocess(self, query, answer, bucket_id, batch_size):
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs, targets_raw = [], [], []

        # pad them if needed, reverse encoder inputs and add GO to decoder.
        batch_source_encoder, batch_source_decoder = [], []
        for batch_i in xrange(batch_size):
            encoder_input = query[batch_i]
            decoder_input = answer[batch_i]

            if data_utils.PAD_ID in encoder_input:
                encoder_input = encoder_input[:encoder_input.index(data_utils.PAD_ID)]
            if data_utils.PAD_ID in decoder_input:
                decoder_input = decoder_input[:decoder_input.index(data_utils.PAD_ID)]

            batch_source_encoder.append(encoder_input)
            batch_source_decoder.append(decoder_input)
            # Encoder inputs are padded and then reversed.
            encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
            if self.config.data_id == 'twitter':
                encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
            else:
                encoder_inputs.append(encoder_pad + encoder_input)

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            de_line = [data_utils.GO_ID] + decoder_input + [data_utils.EOS_ID] + [data_utils.PAD_ID] * (
                decoder_pad_size - 1)
            decoder_inputs.append(de_line[:decoder_size])
            ta_line = decoder_input + [data_utils.EOS_ID] + [data_utils.PAD_ID] * decoder_pad_size
            targets_raw.append(ta_line[:decoder_size])

        # Now we create batch-major vectors from the disc_data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights, batch_targets = [], [], [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(batch_size)], dtype=np.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(batch_size)], dtype=np.int32))

            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(batch_size, dtype=np.float32)
            for batch_idx in xrange(batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                if targets_raw[batch_idx][length_idx] == data_utils.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)

        # Batch targets inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(decoder_size):
            batch_targets.append(
                np.array([targets_raw[batch_idx][length_idx]
                          for batch_idx in xrange(batch_size)], dtype=np.int32))

        return (batch_encoder_inputs, batch_decoder_inputs, batch_weights, batch_source_encoder, batch_source_decoder,
                batch_targets)
