import os
from tqdm import tqdm
import tensorflow as tf
from tensorflow.python.client import timeline
# import cupy as np
import numpy as np
import sys
import time
from six.moves import xrange  # pylint: disable=redefined-builtin
import gen.generator as gens
import reward_model.reward_disc as r_disc
import random
import utils.conf as conf
import utils.data_utils as data_utils
from gen.gen_utils import eval_step
from gen.gen_utils import Gen_Data_Reader
from gen.gen_utils import softmax
from nltk.translate.bleu_score import sentence_bleu
from gen.gen_utils import beam_decoding
import argparse

np.set_printoptions(suppress=True)

gen_config = conf.gen_config
disc_config = conf.disc_config

PARSER = argparse.ArgumentParser(description=None)
PARSER.add_argument('-data_id', '--data_id', default="dailydialog", type=str, help='Dataset: cornell, dialydialog, twitter, MovieTriple')
PARSER.add_argument('-vocab_size', '--vocab_size', default=0, type=int, help='this is the threshold of word frequency')
PARSER.add_argument('-hidden_size', '--hidden_size', default=512, type=int, help='hidden size')
PARSER.add_argument('-ent_weight', '--ent_weight', default=0.001, type=float, help='ent_weight')
PARSER.add_argument('-exp_id', '--exp_id', default=10, type=int, help='exp ID')
PARSER.add_argument('--adv_train', dest='adv', action='store_true', help='adversarial training')
PARSER.add_argument('--no_adv_train', dest='adv', action='store_false', help='supervised training')
PARSER.add_argument('--teacher_forcing', dest='teacher_forcing',action='store_true', help='teacher forcing')
PARSER.add_argument('--no_teacher_forcing', dest='teacher_forcing',action='store_false', help='no teacher forcing')
PARSER.add_argument('--continue_train', dest='continue_train', action='store_true', help='continue training')
PARSER.add_argument('--no_continue_train', dest='continue_train', action='store_false', help='start new training')
PARSER.add_argument('--testing', dest='testing_flag',action='store_true', help='testing and beam search')
PARSER.add_argument('--no_testing', dest='testing_flag',action='store_false', help='just training, no testing')
PARSER.set_defaults(adv=True)
PARSER.set_defaults(teacher_forcing=True)
PARSER.set_defaults(continue_train=False)
PARSER.set_defaults(testing_flag=False)
ARGS = PARSER.parse_args()
print(ARGS)

gen_config.data_id=ARGS.data_id
gen_config.data_dir = os.path.join('./data', ARGS.data_id)
gen_config.emb_dim = ARGS.hidden_size
gen_config.vocab_size=ARGS.vocab_size
gen_config.ent_weight=ARGS.ent_weight
gen_config.exp_id=ARGS.exp_id
gen_config.adv=ARGS.adv
gen_config.teacher_forcing=ARGS.teacher_forcing
gen_config.continue_train=ARGS.continue_train
gen_config.testing=ARGS.testing_flag

disc_config.data_id=ARGS.data_id
disc_config.emb_dim = ARGS.hidden_size
disc_config.data_dir = os.path.join('./data', ARGS.data_id, 'subdata')
disc_config.vocab_size=ARGS.vocab_size
disc_config.ent_weight=ARGS.ent_weight
disc_config.exp_id=ARGS.exp_id
disc_config.adv=ARGS.adv
disc_config.teacher_forcing=ARGS.teacher_forcing
disc_config.continue_train=ARGS.continue_train

# pre train generator
def gen_pre_train():
    gens.train(gen_config)


def mc_sampler_fast(sess=None, gen_model=None, source_inputs=None, source_outputs=None,
                    encoder_inputs=None, decoder_inputs=None, target_weights=None, target_inputs=None, bucket_id=None,
                    disc_model=None, reward_base=0, run_hist = None):
    '''
    :param source_inputs: X with word id
    :param source_outputs: Y^ with word id
    :param encoder_inputs: padded X
    :param decoder_inputs: padded Y
    :param target_weights: padded positions
    :param bucket_id
    :param disc_model: to generate reward for each sampled dialog
    :return: sampled dialogs with MC search, rewards at each position
    '''
    train_query, train_answer = [], []
    query_len = gen_config.buckets[bucket_id][0]
    answer_len = gen_config.buckets[bucket_id][1]

    reward_base = 0.5
    last_pos = 0
    rep_word_punish = []
    for query, answer in zip(source_inputs, source_outputs):
        rep_punish_line = []
        query = query[:query_len] + [int(data_utils.PAD_ID)] * (query_len - len(query) if query_len > len(query) else 0)
        train_query.append(query)
        answer.append(data_utils.EOS_ID)
        # it is necessary to regard EOS as a normal action
        last_pos = max(last_pos, len(answer))
        answer = answer[:answer_len] + [int(data_utils.PAD_ID)] * (
            answer_len - len(answer) if answer_len > len(answer) else 0)
        for x_i, x_w in enumerate(answer):
            if x_w != data_utils.PAD_ID and x_w != 4:
                rep_punish_line.append(max(answer[:x_i].count(x_w) - 2, 0))
            else:
                rep_punish_line.append(0)
        train_answer.append(answer)
        rep_word_punish.append(rep_punish_line)
    train_labels = [0 for _ in source_inputs]
    batch_size = len(train_labels)
    words_positions = []
    rewards_positions = []
    original_rewards = []

    source_inputs_fast = source_inputs * gen_config.beam_size
    source_outputs_fast = source_outputs * gen_config.beam_size
    t1=time.time()
    encoder_inputs_fast = np.array(
        np.array(encoder_inputs).transpose().tolist() * gen_config.beam_size).transpose().tolist()
    decoder_inputs_fast = np.array(
        np.array(decoder_inputs).transpose().tolist() * gen_config.beam_size).transpose().tolist()
    target_weights_fast = np.array(
        np.array(target_weights).transpose().tolist() * gen_config.beam_size).transpose().tolist()
    target_inputs_fast = np.array(
        np.array(target_inputs).transpose().tolist() * gen_config.beam_size).transpose().tolist()

    for pos_id in range(1, last_pos):
        words_beam = []
        reward_each_position = np.zeros((len(train_labels), answer_len))
        q, a, l = sample_relpy_with_x(sess=sess[0], gen_model=gen_model,
                                      source_inputs=source_inputs_fast,
                                      source_outputs=source_outputs_fast,
                                      encoder_inputs=encoder_inputs_fast,
                                      decoder_inputs=decoder_inputs_fast,
                                      target_weights=target_weights_fast,
                                      target_input=target_inputs_fast,
                                      bucket_id=bucket_id,
                                      mc_position=pos_id)
        for i_d in range(gen_config.beam_size):
            start_id = i_d * batch_size
            end_id = (i_d + 1) * batch_size
            words_beam.append([q[start_id:end_id], a[start_id:end_id], l[start_id:end_id]])
        query_ = np.array(q)
        answer_ = np.array(a)
        # a is padded and with EOS_ID
        reward = disc_model.get_reward_forward(sess=sess[1],
                                               bucket_id=bucket_id,
                                               train_query_source=query_,
                                               train_answer_source=answer_,
                                               train_pad_weights=None,
                                               run_hist=run_hist
                                               )

        reward_arr = np.array(reward).reshape(answer_.shape[1], answer_.shape[0])
        reward_arr = np.transpose(reward_arr)  # 5*batch_size x decoder_size
        # print("MC: sample time and rewarding time: %.2f , %.2f" % (t2-t1, t3-t2))
        for i_d in range(gen_config.beam_size):
            reward_each_position += reward_arr[i_d * batch_size: (i_d + 1) * batch_size]

        reward_each_position = reward_each_position / gen_config.beam_size - reward_base
        # this is the average reward of each position
        reward_each_position -= gen_config.repeat_word * np.array(rep_word_punish)
        # reward_each_position = np.clip(reward_each_position, -10, 1)
        reward_each_position = reward_each_position[:, pos_id - 1:]
        reward_each_position = reward_each_position.sum(axis=1)

        words_positions.append(words_beam)
        rewards_positions.append(reward_each_position)

    ### reward for the last action 'EOS',
    reward = disc_model.get_reward_forward(sess=sess[1],
                                           bucket_id=bucket_id,
                                           train_query_source=np.array(train_query),
                                           train_answer_source=np.array(train_answer),
                                           train_pad_weights=None
                                           )
    reward_arr = np.array(reward).reshape(len(train_answer[0]), len(train_answer))
    reward_arr = np.transpose(reward_arr)

    # fill other positions with the whole sentence and overall reward
    for _ in range(0, answer_len - last_pos + 1):
        x = [[train_query, train_answer, train_labels]] * gen_config.beam_size
        words_positions.append(x)
        rewards_positions.append(reward_arr[:, last_pos - 1:last_pos].reshape(-1) - reward_base)
    # mc_reward_adjusted = np.clip(mc_reward_adjusted, -2, 4)
    original_rewards = np.transpose(reward_arr)  # decoder_size X batch_size
    mc_reward_adjusted = rewards_positions
    return words_positions, original_rewards, mc_reward_adjusted


def sample_relpy_with_x(sess=None, gen_model=None, source_inputs=None, source_outputs=None, encoder_inputs=None,
                        decoder_inputs=None, target_weights=None, target_input=None, bucket_id=None, mc_position=0):
    '''
    Sample (X, ^Y) through ^Y ~ G(*|X)
    :param sess:
    :param gen_model: generator model
    :param encoder_inputs: X
    :param decoder_inputs: Y
    :return: sampled reply ^Y
    '''
    train_query, train_answer = [], []
    query_len = gen_config.buckets[bucket_id][0]
    answer_len = gen_config.buckets[bucket_id][1]

    for query in source_inputs:
        query = query[:query_len] + [int(data_utils.PAD_ID)] * (query_len - len(query) if query_len > len(query) else 0)
        train_query.append(query)

    def decoder(num_roll):
        query_list = []
        answer_list = []
        label_list = []
        # "mc_search" in gen_model.step means how you select the word at current position: argmax or sample one
        _, _, sampled_sents, output_logits = gen_model.step(sess,
                                                            encoder_inputs,
                                                            decoder_inputs,
                                                            target_input,
                                                            target_weights,
                                                            bucket_id,
                                                            reward=1,
                                                            forward_only=True, mc_search=True,
                                                            mc_position=mc_position)
        seq_tokens = sampled_sents[0]
        resps = []
        seq_tokens_t = []
        for col in range(len(seq_tokens[0])):
            seq_tokens_t.append([seq_tokens[row][col] for row in range(len(seq_tokens))])

        for seq in seq_tokens_t:
            if data_utils.EOS_ID in seq:
                # EOS_ID should be covered
                resps.append(seq[:seq.index(data_utils.EOS_ID) + 1][:gen_config.buckets[bucket_id][1]])
            else:
                resps.append(seq[:gen_config.buckets[bucket_id][1]])

        for i, output in enumerate(resps):
            output = output[:answer_len] + [data_utils.PAD_ID] * (
                answer_len - len(output) if answer_len > len(output) else 0)
            query_list.append(train_query[i])
            answer_list.append(output)
            label_list.append(0)
        return query_list, answer_list, label_list

    train_query, train_answer, train_label = decoder(1)
    return train_query, train_answer, train_label


# prepare disc_data for discriminator and generator
def disc_train_data(sess, gen_model, source_inputs, source_outputs,
                    encoder_inputs, decoder_inputs, target_weights, target_inputs, bucket_id, mc_search=False):
    train_query, train_answer = [], []
    query_len = gen_config.buckets[bucket_id][0]
    answer_len = gen_config.buckets[bucket_id][1]

    for query, answer in zip(source_inputs, source_outputs):
        query = query[:query_len] + [int(data_utils.PAD_ID)] * (query_len - len(query) if query_len > len(query) else 0)
        train_query.append(query)
        answer.append(data_utils.EOS_ID)  # add tag EOS
        answer = answer[:answer_len] + [int(data_utils.PAD_ID)] * (
            answer_len - len(answer) if answer_len > len(answer) else 0)
        train_answer.append(answer)
    train_labels = [1 for _ in source_inputs]

    num_roll_size = gen_config.num_roll  # num_roll_size * batch_sie: number of sampled trajectories from background distribution
    encoder_inputs_fast = np.array(
        np.array(encoder_inputs).transpose().tolist() * num_roll_size).transpose().tolist()
    decoder_inputs_fast = np.array(
        np.array(decoder_inputs).transpose().tolist() * num_roll_size).transpose().tolist()
    target_weights_fast = np.array(
        np.array(target_weights).transpose().tolist() * num_roll_size).transpose().tolist()
    target_inputs_fast = np.array(
        np.array(target_inputs).transpose().tolist() * num_roll_size).transpose().tolist()

    def decoder(num_roll):
        query_list = []
        answer_list = []
        label_list = []
        traj_prob = []
        for _ in xrange(num_roll):
            # "mc_search" in gen_model.step means how you select the word at current position: argmax or sample one
            _, _, sampled_sents, output_logits = gen_model.step(sess,
                                                                encoder_inputs_fast,
                                                                decoder_inputs_fast,
                                                                target_inputs_fast,
                                                                target_weights_fast,
                                                                bucket_id=bucket_id,
                                                                reward=1,
                                                                forward_only=True,
                                                                mc_search=mc_search,
                                                                mc_position=0)
            resps = []
            seq_tokens = sampled_sents[0]
            seq_tokens_prob = sampled_sents[1]

            seq_tokens_t = []
            logit_prob = []
            for col in range(len(seq_tokens[0])):
                seq_tokens_t.append([seq_tokens[row][col] for row in range(len(seq_tokens))])

            for col in range(len(seq_tokens_prob[0])):
                logit_prob.append([seq_tokens_prob[row][col] for row in range(len(seq_tokens_prob))])

            for ix, seq in enumerate(seq_tokens_t):
                if data_utils.EOS_ID in seq:
                    resps.append(seq[:seq.index(data_utils.EOS_ID) + 1][:gen_config.buckets[bucket_id][1]])
                else:
                    resps.append(seq[:gen_config.buckets[bucket_id][1]])

                traj_prob_each_line = 1.0
                for j, word_id in enumerate(resps[-1]):
                    traj_prob_each_line *= logit_prob[ix][j][word_id]
                traj_prob.append(traj_prob_each_line)

            for i, output in enumerate(resps):
                output = output[:answer_len] + [data_utils.PAD_ID] * (
                    answer_len - len(output) if answer_len > len(output) else 0)
                query_list.append(train_query[i % len(train_query)])
                answer_list.append(output)
                label_list.append(0)
        return query_list, answer_list, label_list, traj_prob

    query_list, answer_list, labels_list, traj_fake_prob = decoder(1)
    train_query += query_list
    train_answer += answer_list
    train_labels += labels_list
    # answer are padded and with EOS_ID

    return train_query, train_answer, train_labels, traj_fake_prob


def train_disc(sess=None, gen_model=None, disc_model=None, train_set=None, bucket_id=None, rev_vocab=None,
               current_step=None, disc_freq=5):
    # 1.Sample (X,Y) from real disc_data
    disc_loss = 0

    def train_disc_single_step(time=0):
        encoder_inputs, decoder_inputs, target_weights, source_inputs, source_outputs, target_inputs = gen_model.get_batch(
            train_set, bucket_id, gen_config.batch_size)

        # 2.Sample (X,Y) and (X, ^Y) through ^Y ~ G(*|X)
        # answers have EOS_ID
        train_query, train_answer, train_labels, traj_fake_prob = disc_train_data(sess[0],
                                                                                  gen_model,
                                                                                  source_inputs,
                                                                                  source_outputs,
                                                                                  encoder_inputs,
                                                                                  decoder_inputs,
                                                                                  target_weights,
                                                                                  target_inputs,
                                                                                  bucket_id,
                                                                                  mc_search=True)
        # train_query = np.transpose(train_query)
        # train_answer = np.transpose(train_answer)

        # 3.Update D using (X, Y ) as positive examples and(X, ^Y) as negative examples
        disc_step_loss = 0
        for _ in range(1):
            # print("inside train_disc")
            # print(len(train_query), len(train_answer), len(train_labels), len(traj_fake_prob))
            disc_step_loss_each, _ = disc_model.update_reward_step(sess=sess[1],
                                                                   bucket_id=bucket_id,
                                                                   train_query=train_query,
                                                                   train_answer=train_answer,
                                                                   train_labels=train_labels,
                                                                   traj_fake_prob=np.array(traj_fake_prob)
                                                                   )
            disc_step_loss += disc_step_loss_each / 1
        return disc_step_loss

    for x_i in range(disc_freq):
        step_loss = train_disc_single_step(time=x_i)
        disc_loss += step_loss / disc_freq

    return disc_loss


# Adversarial Learning for Neural Dialogue Generation
def al_train():
    tf_config = tf.ConfigProto(allow_soft_placement=True, device_count={'GPU': 1})
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    run_time=(run_options, run_metadata)
    np.random.seed(2)
    random.seed(2)
    tf.set_random_seed(2)
    # tf_config.gpu_options.per_process_gpu_memory_fraction = 0.7
    # sess_g = tf.Session(config=tf_config)
    # sess_r = tf.Session(config=tf_config)
    with tf.Session(config=tf_config) as sess_public:
        # sess_pair = (sess_g, sess_r)
        vocab, rev_vocab, test_set, dev_set, train_set = gens.prepare_data(gen_config)
        gen_config.vocab_size = len(rev_vocab)
        print("vocab sizei: {}".format(gen_config.vocab_size))
        for set in train_set:
            print("training set len: ", len(set))
        for set in test_set:
            print("testing set len: ", len(set))

        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(gen_config.buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]
        g1 = tf.Graph()
        with g1.as_default():
            sess_r = tf.Session(config=tf_config, graph=g1)
            disc_model = r_disc.create_model(sess_r, disc_config, disc_config.name_model, vocab)
        g2 = tf.Graph()
        with g2.as_default():
            sess_g = tf.Session(config=tf.ConfigProto(allow_soft_placement=True), graph=g2)
            gen_model = gens.create_model(sess_g, gen_config, forward_only=False, name_scope=gen_config.name_model,
                                      word2id=vocab)
        sess_pair = (sess_g, sess_r)
        # eval_model = eval_disc.create_model(sess, evl_config, evl_config.name_model, vocab)
        current_step = 0
        step_time, disc_loss, gen_loss, t_loss, batch_reward = 0.0, 0.0, 0.0, 0.0, 0.0
        disc_step = 10
        if gen_config.continue_train:
            disc_step = 5
        reward_base = 0
        reward_history = np.zeros(100)
        while True:
            start_time = time.time()
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])
            print("Sampled bucket ID: {}".format(bucket_id))
            # disc_config.max_len = gen_config.buckets[bucket_id][0] + gen_config.buckets[bucket_id][1]
            # b_query, b_gen = train_set[bucket_id], dev_set[bucket_id]
            '''
            if current_step % 10 == 0 and current_step != 0 or (current_step ==0 and gen_config.testing):
                print("==========Evaluate dev set: %d==========" % current_step)
                bleu_score = evaluate_gan(sess=sess_pair,
                                          gen_model=gen_model,
                                          eval_model=None,
                                          gen_config=gen_config,
                                          disc_model=disc_model,
                                          dataset=test_set,
                                          buckets=gen_config.buckets,
                                          rev_vocab=rev_vocab)

                print("Bleu-1 score on dev set: %.4f" % bleu_score[0])
                print("Bleu-2 score on dev set: %.4f" % bleu_score[1])
                print("Bleu-3 score on dev set: %.4f" % bleu_score[2])
            '''
            if gen_config.testing:
                break

            print("==========Update Discriminator: %d==========" % current_step)
            disc_step_loss = train_disc(sess=sess_pair,
                                        gen_model=gen_model,
                                        disc_model=disc_model,
                                        train_set=train_set,
                                        bucket_id=bucket_id,
                                        rev_vocab=rev_vocab,
                                        current_step=current_step,
                                        disc_freq=disc_step)
            disc_step = 5
            disc_loss += disc_step_loss / disc_config.steps_per_checkpoint

            disc_time = time.time()
            print("disc training time %.2f" % (disc_time - start_time))

            print("==========Update Generator: %d==========" % current_step)

            update_gen_data = gen_model.get_batch(train_set, bucket_id, gen_config.batch_size)
            encoder_real, decoder_real, weights_real, source_inputs_real, source_outputs_real, target_real = update_gen_data

            # 2.Sample (X, ^Y) through ^Y ~ G(*|X) with MC
            # answers have no EOS_ID
            sampled_query, sampled_answer, _ = sample_relpy_with_x(sess=sess_g,
                                                                   gen_model=gen_model,
                                                                   source_inputs=source_inputs_real,
                                                                   source_outputs=source_outputs_real,
                                                                   encoder_inputs=encoder_real,
                                                                   decoder_inputs=decoder_real,
                                                                   target_weights=weights_real,
                                                                   target_input=target_real,
                                                                   bucket_id=bucket_id,
                                                                   mc_position=0)
            sample_time = time.time()
            print("sampling time %.2f" % (sample_time - disc_time))
            gen_sampled_batch = gen_model.gen_batch_preprocess(query=sampled_query,
                                                               answer=sampled_answer,
                                                               bucket_id=bucket_id,
                                                               batch_size=gen_config.batch_size)
            # source answers have no EOS_ID
            encoder_sampled, decoder_sampled, weights_sampled, source_inputs_sampled, source_outputs_sampled, target_sampled = gen_sampled_batch

            # 3. MC search to approximate the reward at each position for the sampled reply
            mc_samples, mc_reward, mc_adjusted_word = mc_sampler_fast(sess=sess_pair,
                                                                      gen_model=gen_model,
                                                                      source_inputs=source_inputs_sampled,
                                                                      source_outputs=source_outputs_sampled,
                                                                      encoder_inputs=encoder_sampled,
                                                                      decoder_inputs=decoder_sampled,
                                                                      target_weights=weights_sampled,
                                                                      target_inputs=target_sampled,
                                                                      bucket_id=bucket_id,
                                                                      disc_model=disc_model,
                                                                      reward_base=reward_base,
                                                                      run_hist=run_time)
            reward_history[current_step%100] = np.sum(mc_reward) / np.count_nonzero(mc_reward)
            if current_step<100:
                reward_base = np.sum(reward_history) / (current_step + 1)
            else:
                reward_base = np.sum(reward_history) / 100

            mc_time = time.time()
            print("mc time %.2f" % (mc_time - sample_time))

            batch_reward_step = np.mean(mc_reward[0])
            batch_reward_step_first_line = mc_reward[:, 0]
            # print("step_reward: ", np.mean(mc_reward[-1]))

            # 4.Update G on (X, ^Y ) using mc_reward
            gan_adjusted_loss, gen_step_loss, _, _ = gen_model.step(sess_g,
                                                                    encoder_sampled,
                                                                    decoder_sampled,
                                                                    target_sampled,
                                                                    weights_sampled,
                                                                    bucket_id,
                                                                    forward_only=False,
                                                                    reward=mc_adjusted_word,
                                                                    up_reward=True,
                                                                    debug=True
                                                                    )
            print("step_reward: ", batch_reward_step_first_line)
            print("gen_step_loss: ", gen_step_loss)
            print("gen_step_adjusted_loss: ", gan_adjusted_loss)
            batch_reward += batch_reward_step / gen_config.steps_per_checkpoint
            gen_loss += gen_step_loss / gen_config.steps_per_checkpoint

            gen_time = time.time()
            print("gen update time %.2f" % (gen_time - mc_time))
            print("Gen training time %.2f" % (gen_time - disc_time))

            if gen_config.teacher_forcing:
                print("==========Teacher-Forcing: %d==========" % current_step)
                # encoder_real, decoder_real, weights_real = true_dialog
                reward_god = []
                reward_arr = np.array(weights_real) - 0.0
                for idx in range(len(weights_real)):
                    reward_god.append(np.sum(reward_arr[idx:], axis=0))
                reward_god = np.array(reward_god).tolist()
                t_adjusted_loss, t_step_loss, _, a = gen_model.step(sess_g,
                                                                    encoder_real,
                                                                    decoder_real,
                                                                    target_real,
                                                                    weights_real,
                                                                    bucket_id,
                                                                    reward=reward_god,
                                                                    teacher_forcing=True,
                                                                    forward_only=False)
                t_loss += t_step_loss / gen_config.steps_per_checkpoint
                print("t_step_loss: ", t_step_loss)
                print("t_adjusted_loss", t_adjusted_loss)  # print("normal: ", a)
                teacher_time = time.time()
                print("teacher time %.2f" % (teacher_time - gen_time))

            if current_step % gen_config.steps_per_checkpoint == 0:

                step_time += (time.time() - start_time) / gen_config.steps_per_checkpoint

                print("current_steps: %d, step time: %.4f, disc_loss: %.3f, gen_loss: %.3f, t_loss: %.3f, reward: %.3f"
                      % (current_step, step_time, disc_loss, gen_loss, t_loss, batch_reward))

                if current_step % (gen_config.steps_per_checkpoint * 1) == 0:
                    print("current_steps: %d, save disc model" % current_step)
                    disc_ckpt_dir = os.path.abspath(
                        os.path.join(disc_config.model_dir, 'disc_model',
                                     "data-{}_pre_embed-{}_ent-{}_exp-{}_teacher-{}".format(
                                         disc_config.data_id,
                                         disc_config.pre_embed,
                                         disc_config.ent_weight,
                                         disc_config.exp_id,
                                         disc_config.teacher_forcing)))
                    if not os.path.exists(disc_ckpt_dir):
                        os.makedirs(disc_ckpt_dir)
                    disc_model_path = os.path.join(disc_ckpt_dir, "disc.model")
                    disc_model.saver.save(sess_r, disc_model_path, global_step=disc_model.global_step)

                    print("current_steps: %d, save gen model" % current_step)
                    gen_ckpt_dir = os.path.abspath(
                        os.path.join(gen_config.model_dir, 'gen_model',
                                     "data-{}_pre_embed-{}_ent-{}_exp-{}_teacher-{}".format(
                                         gen_config.data_id,
                                         gen_config.pre_embed,
                                         gen_config.ent_weight,
                                         gen_config.exp_id,
                                         gen_config.teacher_forcing)))
                    if not os.path.exists(gen_ckpt_dir):
                        os.makedirs(gen_ckpt_dir)
                    gen_model_path = os.path.join(gen_ckpt_dir, "gen.model")
                    gen_model.saver.save(sess_g, gen_model_path, global_step=gen_model.global_step)

                    step_time, disc_loss, gen_loss, t_loss, batch_reward = 0.0, 0.0, 0.0, 0.0, 0.0
                    sys.stdout.flush()

            current_step += 1


def main(_):
    # step_1 training gen model
    # gen_pre_train()
    al_train()


if __name__ == "__main__":
    tf.app.run()
