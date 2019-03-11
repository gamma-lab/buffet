from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import pickle
import heapq
import tensorflow.python.platform
from nltk.translate.bleu_score import sentence_bleu


import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tqdm import tqdm

import utils.data_utils as data_utils
import utils.conf as conf
import gen.gen_model as seq2seq_model
from tensorflow.python.platform import gfile
from gen.gen_utils import Gen_Data_Reader

sys.path.append('../utils')


# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.


def read_data(config, source_path, target_path, max_size=None, unique_list=[]):
    data_set = [[] for _ in config.buckets]
    with gfile.GFile(source_path, mode="r") as source_file:
        with gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 100000 == 0:
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                # target_ids.append(data_utils.EOS_ID)
                # if [source_ids, target_ids] not in unique_list:
                for bucket_id, (source_size, target_size) in enumerate(
                                config.buckets):  # [bucket_id, (source_size, target_size)]
                    if 3 < len(source_ids) < source_size - 2 and 3 < len(target_ids) < target_size - 2:
                        data_set[bucket_id].append([source_ids, target_ids])
                            # store unique training samples.
                            # unique_list.append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set, unique_list


def create_model(session, gen_config, forward_only, name_scope, word2id, initializer=None):
    """Create translation model and initialize or load parameters in session."""
    with tf.variable_scope(name_or_scope=name_scope, initializer=initializer):
        model = seq2seq_model.Seq2SeqModel(gen_config, name_scope=name_scope, forward_only=forward_only)
        if not gen_config.adv:
            gen_ckpt_dir = os.path.abspath(os.path.join(gen_config.data_dir, 'gen_model', "checkpoints"))
        else:
            gen_ckpt_dir = os.path.abspath(os.path.join(gen_config.model_dir, 'gen_model',
                                                                "data-{}_pre_embed-{}_exp-{}".format(
                                                                    gen_config.data_id,
                                                                    gen_config.pre_embed,
                                                                    1)))
        # gen_config.continue_train==True will overwrite the previous gen_ckpt_dir and continue adv-training
        if gen_config.continue_train:
            gen_ckpt_dir = os.path.abspath(
                        os.path.join(gen_config.model_dir, 'gen_model',
                                     "data-{}_pre_embed-{}_ent-{}_exp-{}_teacher-{}".format(
                                         gen_config.data_id,
                                         gen_config.pre_embed,
                                         gen_config.ent_weight,
                                         gen_config.exp_id,
                                         gen_config.teacher_forcing)))

        print("check model path: %s" % gen_ckpt_dir)
        ckpt = tf.train.get_checkpoint_state(gen_ckpt_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading Gen model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(session, ckpt.model_checkpoint_path)
            load_embeddings_generator(session, name_scope, word2id, gen_config.word_embedding, True)
            # reset_lr = model.learning_rate.assign(gen_config.learning_rate)
            # session.run(reset_lr)
        else:
            print("Create Gen model with fresh parameters.")
            gen_global_variables = [gv for gv in tf.global_variables() if name_scope in gv.name]
            session.run(tf.variables_initializer(gen_global_variables))
            print("Finished Creating Gen model with fresh parameters.")
            if gen_config.pre_embed:
                load_embeddings_generator(session, name_scope, word2id, gen_config.word_embedding, False)
        return model


def load_embeddings_generator(sess, name_scope, word2id, embedding_size, pre_trained=False):
    """ Initialize embeddings with pre-trained word2vec vectors
            Will modify the embedding weights of the current loaded model
            Uses the GoogleNews pre-trained values (path hardcoded)
            """
    # Fetch embedding variables from model
    embedding_path = '/home/zli1/dialogue-gan/data/embedding/GoogleNews-vectors-negative300.bin'
    # with tf.variable_scope(name_scope+"/embedding_rnn_seq2seq/rnn/embedding_wrapper", reuse=True):
    variables = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
    with tf.variable_scope("embedding_attention_seq2seq/rnn/embedding_wrapper", reuse=True):
        em_in = tf.get_variable("embedding")
    # print("em_in finished")
    with tf.variable_scope("embedding_attention_seq2seq", reuse=True):
        em_out = tf.get_variable("embedding")
    # variables.remove(em_in)
    # variables.remove(em_out)

    if pre_trained:
        return
    # New model, we load the pre-trained word2vec data and initialize embeddings
    embeddings_path = embedding_path
    embeddings_format = os.path.splitext(embeddings_path)[1][1:]
    print("Loading pre-trained word embeddings from %s " % embeddings_path)
    with open(embeddings_path, "rb") as f:
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * vector_size
        initW = np.random.uniform(-0.25, 0.25, (len(word2id), vector_size))
        for line in tqdm(range(vocab_size)):
            word = []
            while True:
                ch = f.read(1)
                if ch == b' ':
                    word = b''.join(word).decode('utf-8')
                    break
                if ch != b'\n':
                    word.append(ch)
            if word in word2id:
                if embeddings_format == 'bin':
                    vector = np.fromstring(f.read(binary_len), dtype='float32')
                elif embeddings_format == 'vec':
                    vector = np.fromstring(f.readline(), sep=' ', dtype='float32')
                else:
                    raise Exception("Unkown format for embeddings: %s " % embeddings_format)
                initW[word2id[word]] = vector
            else:
                if embeddings_format == 'bin':
                    f.read(binary_len)
                elif embeddings_format == 'vec':
                    f.readline()
                else:
                    raise Exception("Unkown format for embeddings: %s " % embeddings_format)

    # PCA Decomposition to reduce word2vec dimensionality
    if embedding_size < vector_size:
        U, s, Vt = np.linalg.svd(initW, full_matrices=False)
        S = np.zeros((vector_size, vector_size), dtype=complex)
        S[:vector_size, :vector_size] = np.diag(s)
        initW = np.dot(U[:, :embedding_size], S[:embedding_size, :embedding_size])

    # Initialize input and output embeddings
    sess.run(em_in.assign(initW))
    sess.run(em_out.assign(initW))


def prepare_data(gen_config):
    train_path = os.path.join(gen_config.data_dir, "train")
    test_path = os.path.join(gen_config.data_dir, "test")
    dev_path = os.path.join(gen_config.data_dir, "dev")
    voc_file_path = [train_path + ".answer", train_path + ".query", test_path + ".answer", test_path + ".query",
                     dev_path + ".answer", dev_path + ".query"]
    vocab_path = os.path.join(gen_config.data_dir, "vocab%d.all" % gen_config.vocab_size)
    data_utils.create_vocabulary(vocab_path, voc_file_path, gen_config.vocab_size)
    # vocab_path = os.path.join(gen_config.data_dir, "vocab%d.all" % 30000)
    # TODO: change 30000 to 2500

    vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

    # print("Preparing Chitchat gen_data in %s" % gen_config.train_dir)
    train_query, train_answer, dev_query, dev_answer, test_query, test_answer = data_utils.prepare_chitchat_data(
        gen_config.data_dir, vocab, gen_config.vocab_size)
    # train_query, train_answer, dev_query, dev_answer = data_utils.prepare_chitchat_data_OpenSub(gen_config.data_dir)

    # Read disc_data into buckets and compute their sizes.
    print("Reading development and training gen_data (limit: %d)."
          % gen_config.max_train_data_size)

    unique_list = []
    train_set, unique_list = read_data(gen_config, train_query, train_answer,  unique_list=unique_list)
    dev_set, unique_list = read_data(gen_config, dev_query, dev_answer, unique_list=unique_list)
    test_set, unique_list = read_data(gen_config, test_query, test_answer, unique_list=unique_list)

    return vocab, rev_vocab, test_set, dev_set, train_set


def softmax(x):
    prob = np.exp(x) / np.sum(np.exp(x), axis=0)
    return prob


def train(gen_config):
    vocab, rev_vocab, test_set, dev_set, train_set = prepare_data(gen_config)
    for b_set in train_set:
        print("b_set: ", len(b_set))

    with tf.Session() as sess:
        model = create_model(sess, gen_config, forward_only=False, name_scope=gen_config.name_model, word2id=vocab)

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        # previous_losses = []

        gen_loss_summary = tf.Summary()
        gen_writer = tf.summary.FileWriter(gen_config.tensorboard_dir, sess.graph)

        while True:
            bucket_id = current_step % len(gen_config.buckets)
            gen_data_reader = Gen_Data_Reader(train_set[bucket_id])
            batch_number = gen_data_reader.get_batch_num(gen_config.batch_size)
            start_time = time.time()
            loss_bucket = 0
            # Get a batch and make a step.
            for batch_id in range(batch_number):
                train_batch = gen_data_reader.generate_training_batch(gen_config.batch_size)
                encoder_inputs, decoder_inputs, target_weights, batch_source_encoder, batch_source_decoder, target_inputs = model.get_batch(
                    train_batch, bucket_id, gen_config.batch_size, type=3)
                _, step_loss, _, _ = model.step(sess, encoder_inputs, decoder_inputs, target_inputs, target_weights,
                                                bucket_id,
                                                forward_only=False)
                loss_bucket += step_loss / batch_number

            loss += loss_bucket / gen_config.steps_per_checkpoint
            step_time += (time.time() - start_time) / gen_config.steps_per_checkpoint
            print("current step: %d, bucket id: %d, loss: %.4f " % (current_step, bucket_id, loss_bucket))

            # Once in a while, we save checkpoint, print statistics, and run evals.
            # if current_step % gen_config.steps_per_checkpoint == 0:
            if current_step % 10 == 0:
                bucket_value = gen_loss_summary.value.add()
                bucket_value.tag = gen_config.name_loss
                bucket_value.simple_value = float(loss)
                gen_writer.add_summary(gen_loss_summary, int(model.global_step.eval()))

                # Print statistics for the previous epoch.
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print("global step %d learning rate %.4f step-time %.2f perplexity "
                      "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                step_time, perplexity))
                # Decrease learning rate if no improvement was seen over last 3 times.
                # if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                #     sess.run(model.learning_rate_decay_op)
                # previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.

                evaluate_generator(sess=sess,
                                   gen_model=model,
                                   gen_config=gen_config,
                                   dataset=dev_set,
                                   buckets=gen_config.buckets,
                                   rev_vocab=rev_vocab)

                if current_step % (gen_config.steps_per_checkpoint * 1) == 0:
                    print("current_step: %d, save model" % (current_step))
                    gen_ckpt_dir = os.path.abspath(os.path.join(gen_config.model_dir, 'gen_model',
                                                                "data-{}_pre_embed-{}_exp-{}".format(
                                                                    gen_config.data_id,
                                                                    gen_config.pre_embed,
                                                                    gen_config.exp_id)))
                    if not os.path.exists(gen_ckpt_dir):
                        os.makedirs(gen_ckpt_dir)
                    checkpoint_path = os.path.join(gen_ckpt_dir, "chitchat.model")
                    model.saver.save(sess, checkpoint_path, global_step=model.global_step)

                step_time, loss = 0.0, 0.0

                # Run evals on development set and print their perplexity.
                # for bucket_id in xrange(len(gen_config.buckets)):
                #   encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                #       dev_set, bucket_id)
                #   _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                #                                target_weights, bucket_id, True)
                #   eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                #   print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
                sys.stdout.flush()
            current_step += 1


def evaluate_generator(sess=None, gen_model=None, gen_config=None, dataset=None, buckets=None,
                       rev_vocab=None):
    buckets_num = len(buckets)
    sel_bid = np.random.randint(0, buckets_num)

    bleu_weights_1 = (1, 0, 0, 0)
    bleu_weights_2 = (0, 1, 0, 0)
    bleu_weights_3 = (0, 0, 1, 0)

    def bleu_gan(ref, pred):
        score_batch_1 = 0
        score_batch_2 = 0
        score_batch_3 = 0
        length = len(pred)
        for x, y in zip(ref, pred):
            if data_utils.PAD_ID in x:
                x_nopad = x[:x.index(data_utils.PAD_ID)]
            else:
                x_nopad = x[:]
            if data_utils.PAD_ID in y:
                y_nopad = y[:y.index(data_utils.PAD_ID)]
            else:
                y_nopad = y[:]
            score_batch_1 += sentence_bleu([x_nopad], y_nopad, bleu_weights_1) / length
            score_batch_2 += sentence_bleu([x_nopad], y_nopad, bleu_weights_2) / length
            score_batch_3 += sentence_bleu([x_nopad], y_nopad, bleu_weights_3) / length
        return score_batch_1, score_batch_2, score_batch_3

    bleu_score_1 = 0
    bleu_score_2 = 0
    bleu_score_3 = 0
    bleu_ins_counter = 0

    for bucket_id in range(buckets_num):
        gen_data_reader = Gen_Data_Reader(dataset[bucket_id])
        batch_number = gen_data_reader.get_batch_num(gen_config.batch_size)
        for batch_id in range(batch_number):
            train_batch = gen_data_reader.generate_training_batch(gen_config.batch_size)
            encoder_inputs_eval, decoder_inputs_eval, target_weights_eval, source_inputs_eval, source_outputs_eval, target_input_eval = gen_model.get_batch(
                train_batch, bucket_id, gen_config.batch_size, type=3)

            # 2.Sample (X,Y) and (X, ^Y) through ^Y ~ G(*|X)
            train_query_eval, train_answer_eval, train_labels_eval = sample_testing(sess, gen_model,
                                                                                    source_inputs_eval,
                                                                                    source_outputs_eval,
                                                                                    encoder_inputs_eval,
                                                                                    decoder_inputs_eval,
                                                                                    target_weights_eval,
                                                                                    target_input_eval,
                                                                                    bucket_id,
                                                                                    mc_search=False)
            bleu_batch_num = int(len(train_answer_eval) / 2)
            b1, b2, b3 = bleu_gan(train_answer_eval[:bleu_batch_num], train_answer_eval[bleu_batch_num:])
            bleu_score_1 += b1
            bleu_score_2 += b2
            bleu_score_3 += b3
            bleu_ins_counter += 1

            if batch_id == 0 and bucket_id == sel_bid:
                print("train_query: ", len(train_query_eval))
                print("train_answer: ", len(train_answer_eval))
                print("train_labels: ", len(train_labels_eval))
                for i in xrange(len(train_query_eval)):
                    print(" ".join(
                        [tf.compat.as_str(rev_vocab[output].encode('ascii', 'ignore').decode('ascii')) for output in
                         train_query_eval[i] if output > 0]))
                    print(" ".join(
                        [tf.compat.as_str(rev_vocab[output].encode('ascii', 'ignore').decode('ascii')) for output in
                         train_answer_eval[i] if output > 0]))
                    print("label: ", train_labels_eval[i])
    print("Bleu-1 score on dev set: %.4f" % (bleu_score_1 / bleu_ins_counter))
    print("Bleu-2 score on dev set: %.4f" % (bleu_score_2/bleu_ins_counter))
    print("Bleu-3 score on dev set: %.4f" % (bleu_score_3/bleu_ins_counter))


def sample_testing(sess, gen_model, source_inputs, source_outputs,
                   encoder_inputs, decoder_inputs, target_weights,
                   target_inputs, bucket_id, mc_search=False):
    train_query, train_answer = [], []
    query_len = gen_config.buckets[bucket_id][0]
    answer_len = gen_config.buckets[bucket_id][1]

    for query, answer in zip(source_inputs, source_outputs):
        query = query[:query_len] + [int(data_utils.PAD_ID)] * (query_len - len(query) if query_len > len(query) else 0)
        train_query.append(query)
        answer = answer[:-1]  # del tag EOS
        answer = answer[:answer_len] + [int(data_utils.PAD_ID)] * (
            answer_len - len(answer) if answer_len > len(answer) else 0)
        train_answer.append(answer)
    train_labels = [1 for _ in source_inputs]

    def decoder(num_roll):
        query_list = []
        answer_list = []
        label_list = []
        for _ in xrange(num_roll):
            # "mc_search" in gen_model.step means how you select the word at current position: argmax or sample one
            _, _, sampled_sents, output_logits = gen_model.step(sess,
                                                                encoder_inputs,
                                                                decoder_inputs,
                                                                target_inputs,
                                                                target_weights,
                                                                bucket_id=bucket_id,
                                                                reward=1,
                                                                forward_only=True,
                                                                mc_search=mc_search,
                                                                mc_position=0)
            seq_tokens = []
            resps = []

            if not mc_search:
                # each position in the sent
                # len(output_logit) == Decoder_length
                for seq in output_logits:
                    row_token = []
                    # each line in the batch
                    for t in seq:
                        row_token.append(int(np.argmax(t, axis=0)))
                    seq_tokens.append(row_token)
            elif mc_search:
                seq_tokens = sampled_sents

            seq_tokens_t = []
            for col in range(len(seq_tokens[0])):
                seq_tokens_t.append([seq_tokens[row][col] for row in range(len(seq_tokens))])

            for seq in seq_tokens_t:
                if data_utils.EOS_ID in seq:
                    resps.append(seq[:seq.index(data_utils.EOS_ID)][:gen_config.buckets[bucket_id][1]])
                else:
                    resps.append(seq[:gen_config.buckets[bucket_id][1]])

            for i, output in enumerate(resps):
                output = output[:answer_len] + [data_utils.PAD_ID] * (
                    answer_len - len(output) if answer_len > len(output) else 0)
                # train_query.append(train_query[i])
                query_list.append(train_query[i])
                # train_answer.append(output)
                answer_list.append(output)
                # train_labels.append(0)
                label_list.append(0)
        return query_list, answer_list, label_list

    query_list, answer_list, labels_list = decoder(1)
    train_query += query_list
    train_answer += answer_list
    train_labels += labels_list

    return train_query, train_answer, train_labels


def main(_):
    gen_config = conf.gen_config
    train(gen_config)


if __name__ == "__main__":
    tf.app.run()
