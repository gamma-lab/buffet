import tensorflow as tf
import time
import numpy as np
# import cupy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.python.ops import math_ops
import utils.data_utils as data_utils
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.ops import embedding_ops
from gen.tf_utils import EmbeddingWrapper as EmbeddingWrapper_GPU

linear = core_rnn_cell._Linear  # pylint: disable=protected-access


class Hier_reward_model(object):
    def __init__(self, config, name_scope, dtype=tf.float32):
        # with tf.variable_scope(name_or_scope=scope_name):
        with tf.device("/gpu:0"):
            emb_dim = config.embed_dim
            word_embedding = config.word_embedding
            num_layers = config.num_layers
            vocab_size = config.vocab_size
            buckets = config.buckets
            self.learning_rate = tf.Variable(float(config.learning_rate), trainable=False, dtype=dtype)
            self.global_step = tf.Variable(initial_value=0, trainable=False)

            self.query = []
            self.answer = []
            self.weight = []
            for i in range(buckets[-1][0]):
                self.query.append(tf.placeholder(dtype=tf.int32, shape=[None], name="query{0}".format(i)))
            for i in xrange(buckets[-1][1]):
                self.answer.append(tf.placeholder(dtype=tf.int32, shape=[None], name="answer{0}".format(i)))
            for i in xrange(buckets[-1][1]):
                self.weight.append(tf.placeholder(dtype=tf.float32, shape=[None], name="weight{0}".format(i)))

            self.traj_ip_weight = tf.placeholder(dtype=tf.float32, shape=[None], name="traj_weight")

            # self.target = tf.placeholder(dtype=tf.int64, shape=[None], name="target")

            def create_rnn_cell():
                # encoDecoCell = tf.contrib.rnn.GRUCell(  # Or GRUCell, LSTMCell(args.hiddenSize)
                encoDecoCell = tf.nn.rnn_cell.GRUCell(  # Or GRUCell, LSTMCell(args.hiddenSize)
                    emb_dim,
                )
                encoDecoCell = tf.contrib.rnn.DropoutWrapper(
                    encoDecoCell,
                    input_keep_prob=1.0,
                    output_keep_prob=config.keep_prob
                )
                return encoDecoCell

            # '''
            encoder_mutil = tf.contrib.rnn.MultiRNNCell(
                [create_rnn_cell() for _ in range(num_layers)],
            )
            # '''
            query_encoder_emb = EmbeddingWrapper_GPU(encoder_mutil, embedding_classes=vocab_size,
                                                                embedding_size=word_embedding)

            context_multi = tf.contrib.rnn.MultiRNNCell(
                [create_rnn_cell() for _ in range(1)],
            )

            self.b_query_state = []
            self.b_answer_state = []
            self.b_state = []
            self.b_reward = []
            self.b_loss = []
            self.b_train_op = []
            self.b_traj_reward = []
            # with tf.name_scope('structure'):
            for i, bucket in enumerate(buckets):
                state_list = []
                reward_list = []
                with tf.variable_scope(name_or_scope="Hier_RNN_encoder", reuse=True if i > 0 else None) as scope:
                    query_output, query_state = tf.contrib.rnn.static_rnn(query_encoder_emb,
                                                                          inputs=self.query[:bucket[0]],
                                                                          dtype=tf.float32)
                    self.b_query_state.append(query_state)

                with tf.variable_scope("Hier_RNN_encoder/rnn/embedding_wrapper", reuse=True):
                    embed_in = tf.get_variable("embedding")
                    emb_answer = [
                        embedding_ops.embedding_lookup(embed_in, ix) for ix in self.answer[:bucket[1]]]

                with tf.variable_scope(name_or_scope="Hier_RNN_context", reuse=True if i > 0 else None) as var_scope:
                    '''
                    utilize the state from last step which record the hidden state of each encoding step
                    '''
                    query_state_history = query_state[-1]
                    context_action_history = []
                    for j in range(0, bucket[1]):
                        if j > 0:
                            var_scope.reuse_variables()
                        action = emb_answer[j]
                        emb_proj_w = tf.get_variable("embd_project_w", [word_embedding, emb_dim], dtype=tf.float32,
                                                     initializer=tf.random_normal_initializer(stddev=0.1))
                        emb_proj_b = tf.get_variable("embd_project_b", [emb_dim], dtype=tf.float32,
                                                     initializer=tf.random_normal_initializer(stddev=0.1))
                        projected = tf.matmul(action, emb_proj_w) + emb_proj_b
                        context_action_history.append(projected)

                with tf.variable_scope(name_or_scope="Reward_concat_layer", reuse=True if i > 0 else None) as var_scope:
                    context_input = [query_state_history] + context_action_history
                    output, state = tf.contrib.rnn.static_rnn(context_multi, context_input, dtype=tf.float32)
                    for j in range(0, bucket[1]):
                        state_action_pair = [output[j], context_action_history[j]]
                        state_list.append(state_action_pair)
                self.b_state.append(state_list)

                with tf.variable_scope("Softmax_layer_and_output", reuse=True if i > 0 else None) as var_scope:
                    for j in range(0, bucket[1]):
                        if j > 0:
                            var_scope.reuse_variables()
                        softmax_w1_s = tf.get_variable("softmax_w1_s", [emb_dim, 100], dtype=tf.float32,
                                                       initializer=tf.random_normal_initializer(stddev=0.1))
                        softmax_b1_s = tf.get_variable("softmax_b1_s", 100, dtype=tf.float32,
                                                       initializer=tf.random_normal_initializer(stddev=0.1))
                        softmax_w1_a = tf.get_variable("softmax_w1_a", [emb_dim, 100], dtype=tf.float32,
                                                       initializer=tf.random_normal_initializer(stddev=0.1))
                        softmax_b1_a = tf.get_variable("softmax_b1_a", 100, dtype=tf.float32,
                                                       initializer=tf.random_normal_initializer(stddev=0.1))
                        s_1 = tf.matmul(state_list[j][0], softmax_w1_s) + softmax_b1_s
                        a_1 = tf.matmul(state_list[j][1], softmax_w1_a) + softmax_b1_a
                        s_a_1 = tf.concat([s_1, a_1], 1)

                        softmax_w3 = tf.get_variable("softmax_w3", [200, 100], dtype=tf.float32,
                                                     initializer=tf.random_normal_initializer(stddev=0.1))
                        softmax_b3 = tf.get_variable("softmax_b3", 100, dtype=tf.float32,
                                                     initializer=tf.random_normal_initializer(stddev=0.1))
                        softmax_w4 = tf.get_variable("softmax_w4", [100, 50], dtype=tf.float32,
                                                     initializer=tf.random_normal_initializer(stddev=0.1))
                        softmax_b4 = tf.get_variable("softmax_b4", 50, dtype=tf.float32,
                                                     initializer=tf.random_normal_initializer(stddev=0.1))
                        softmax_w5 = tf.get_variable("softmax_w5", [50, 1], dtype=tf.float32,
                                                     initializer=tf.random_normal_initializer(stddev=0.1))
                        softmax_b5 = tf.get_variable("softmax_b5", 1, dtype=tf.float32,
                                                     initializer=tf.random_normal_initializer(stddev=0.1))

                        logits_mid1 = tf.matmul(s_a_1, softmax_w3) + softmax_b3
                        logits_mid2 = tf.matmul(logits_mid1, softmax_w4) + softmax_b4
                        logits = tf.matmul(logits_mid2, softmax_w5) + softmax_b5

                        reward = tf.nn.sigmoid(logits)
                        reward = tf.reshape(reward, [-1])
                        # print(reward.get_shape())
                        reward = tf.multiply(reward, self.weight[j])
                        # print(self.weight[j].get_shape())
                        reward_list.append(reward)
                self.b_reward.append(reward_list)

                with tf.name_scope("loss"):
                    traj_reward = math_ops.add_n(reward_list)
                    loss = tf.multiply(traj_reward, self.traj_ip_weight)
                    # mean_loss = tf.reduce_mean(loss)
                    mean_loss = tf.reduce_sum(loss)
                    self.b_loss.append(mean_loss)
                    self.b_traj_reward.append(traj_reward)

                with tf.name_scope("gradient_descent"):
                    # '''
                    optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
                    '''
                    optimizer = tf.train.AdamOptimizer(
                        learning_rate=self.learning_rate,
                        beta1=0.9,
                        beta2=0.999,
                        epsilon=1e-08
                    )
                    '''
                    gradients, variables = zip(*optimizer.compute_gradients(mean_loss))
                    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                    train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)
                    self.b_train_op.append(train_op)

            all_variables = [v for v in tf.global_variables() if name_scope in v.name]
            self.saver = tf.train.Saver(all_variables)

    def get_batch(self, query_batch, answer_batch):
        """
        :param query_batch: padded without EOS
        :param answer_batch: padded without EOS
        :param label_batch: real dialog or fake dialog
        :return:
        """
        bucket = [len(query_batch[0]), len(answer_batch[0])]
        train_query, train_answer = [], []
        weights_all = []
        for query_, answer_ in zip(query_batch, answer_batch):
            if data_utils.PAD_ID in answer_:
                real_value_len = answer_.tolist().index(data_utils.PAD_ID)
            else:
                real_value_len = bucket[1]
            weight_ = [1] * real_value_len + [0] * (bucket[1] - real_value_len)
            weights_all.append(weight_)
            train_query.append(query_)
            train_answer.append(answer_)
        return train_query, train_answer, weights_all

    def get_reward_forward(self, sess=None, bucket_id=None, train_query_source=None, train_answer_source=None,
                           train_pad_weights=None, run_hist=(None, None)):
        if train_pad_weights is None:
            train_query, train_answer, train_weights = self.get_batch(train_query_source, train_answer_source)
        else:
            train_query, train_answer, train_weights = train_query_source, train_answer_source, train_pad_weights

        train_query_ = np.transpose(train_query)
        train_answer_ = np.transpose(train_answer)
        train_weights_ = np.transpose(train_weights)

        feed_dict = {}
        for i in xrange(len(train_query_)):
            feed_dict[self.query[i].name] = train_query_[i]

        for i in xrange(len(train_answer_)):
            feed_dict[self.answer[i].name] = train_answer_[i]

        for i in xrange(len(train_weights_)):
            feed_dict[self.weight[i].name] = train_weights_[i]
        fetches = self.b_reward[bucket_id]
        reward = sess.run(fetches, feed_dict)  # shape(reward) = decoder_size X batch_size
        return reward

    def get_ip_weights(self, sess=None, bucket_id=None, g_prob=None, train_query_fake=None, train_answer_fake=None,
                       train_pad_weights_fake=None):
        reward = self.get_reward_forward(sess, bucket_id, train_query_fake, train_answer_fake, train_pad_weights_fake)
        reward_batch = np.sum(np.array(reward).reshape(len(train_answer_fake[0]), len(train_answer_fake)), axis=0)
        traj_prob = g_prob
        # traj_prob = 1.0 / len(train_query_fake) * g_prob
        ip_weights = np.exp(reward_batch) / traj_prob
        ip_weights = ip_weights / np.sum(ip_weights)

        return ip_weights

    def update_reward_step(self, sess=None, bucket_id=None, train_query=None, train_answer=None, train_labels=None,
                           traj_fake_prob=None):
        '''
        :param sess:
        :param bucket_id:
        :param train_query: make sure that real dialogs are in front of fake ones
        :param train_answer:
        :param train_labels:
        :return:
        loss: training loss
        traj_reward: the reward for each traj
        '''
        train_query_real, train_answer_real = [], []
        train_query_fake, train_answer_fake = [], []

        for query_, answer_, label_ in zip(train_query, train_answer, train_labels):
            if label_ == 1:
                train_query_real.append(query_)
                train_answer_real.append(answer_)
            else:
                train_query_fake.append(query_)
                train_answer_fake.append(answer_)
        train_query_real, train_answer_real, train_pad_weights_real = self.get_batch(np.array(train_query_real),
                                                                                     np.array(train_answer_real))
        train_ip_weights_real = [-1.0 / len(train_query_real)] * len(train_query_real)
        train_query_fake, train_answer_fake, train_pad_weights_fake = self.get_batch(np.array(train_query_fake),
                                                                                     np.array(train_answer_fake))
        '''
        train_ip_weights_fake = self.get_ip_weights(sess=sess,
                                                    bucket_id=bucket_id,
                                                    g_prob=traj_fake_prob,
                                                    train_query_fake=train_query_fake,
                                                    train_answer_fake=train_answer_fake,
                                                    train_pad_weights_fake=train_pad_weights_fake)
        train_ip_weights_all = train_ip_weights_real + train_ip_weights_fake.reshape(-1).tolist()
        '''
        train_ip_weights_fake = [1.0 / len(train_query_fake)] * len(train_query_fake)
        train_ip_weights_all = train_ip_weights_real + train_ip_weights_fake
        #'''
        train_query_all = train_query_real + train_query_fake
        train_answer_all = train_answer_real + train_answer_fake
        train_pad_weights_all = train_pad_weights_real + train_pad_weights_fake

        train_query_ = np.transpose(train_query_all)
        train_answer_ = np.transpose(train_answer_all)
        train_weights_ = np.transpose(train_pad_weights_all)

        feed_dict = {}
        for i in xrange(len(train_query_)):
            feed_dict[self.query[i].name] = train_query_[i]
        for i in xrange(len(train_answer_)):
            feed_dict[self.answer[i].name] = train_answer_[i]
        for i in xrange(len(train_weights_)):
            feed_dict[self.weight[i].name] = train_weights_[i]
        feed_dict[self.traj_ip_weight.name] = train_ip_weights_all

        fetches = [self.b_train_op[bucket_id], self.b_loss[bucket_id], self.b_traj_reward[bucket_id]]
        _, loss, traj_reward = sess.run(fetches, feed_dict)

        return loss, traj_reward
