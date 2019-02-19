import tensorflow as tf
import numpy as np
import os
from reward_model.reward_func_model import Hier_reward_model
import utils.conf as conf
from gen.generator import prepare_data
from gen.generator import read_data
import utils.data_utils as data_utils
from gen.gen_utils import Gen_Data_Reader
import sys
from tqdm import tqdm
import argparse

sys.path.append("../utils")

def parse_args():
    PARSER = argparse.ArgumentParser(description=None)
    PARSER.add_argument('-data_id', '--data_id', default="dailydialog", type=str, help='Dataset: cornell, dialydialog, twitter, MovieTriple')
    PARSER.add_argument('-vocab_size', '--vocab_size', default=10000, type=int, help='vocabulary table size')
    PARSER.add_argument('-hidden_size', '--hidden_size', default=512, type=int, help='hidden size of LSTM')
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
    PARSER.add_argument('-beam_file', '--beam_file', default="", type=str, help='Beam file name')
    PARSER.add_argument('-beam_dir', '--beam_dir', default="", type=str, help='beam dir')
    PARSER.set_defaults(adv=True)
    PARSER.set_defaults(teacher_forcing=True)
    PARSER.set_defaults(continue_train=False)
    PARSER.set_defaults(testing_flag=False)
    ARGS = PARSER.parse_args()
    print(ARGS)
    return ARGS

def param_create(ARGS):
    disc_config = conf.disc_config
    disc_config.data_id=ARGS.data_id
    disc_config.data_dir = os.path.join('../dialogue-gan/data', ARGS.data_id)
    disc_config.vocab_size=ARGS.vocab_size
    disc_config.ent_weight=ARGS.ent_weight
    disc_config.exp_id=ARGS.exp_id
    disc_config.adv=ARGS.adv
    disc_config.teacher_forcing=ARGS.teacher_forcing
    disc_config.continue_train=ARGS.continue_train
    disc_config.beam_file = ARGS.beam_file
    disc_config.beam_dir = ARGS.beam_dir
    return disc_config

def create_model(sess, config, name_scope, word2id, initializer=None):
    with tf.variable_scope(name_or_scope=name_scope, initializer=initializer):
        model = Hier_reward_model(config=config, name_scope=name_scope)
        if not config.adv:
            disc_ckpt_dir = os.path.abspath(os.path.join(config.model_dir, 'disc_model', "checkpoints"))
        else:
            disc_ckpt_dir = os.path.abspath(os.path.join(config.model_dir, 'disc_model',
                                                         "data-{}_pre_embed-{}_exp-{}".format(
                                                             config.data_id,
                                                             config.pre_embed,
                                                             config.exp_id)))
        if config.continue_train:
            disc_ckpt_dir = os.path.abspath(
                os.path.join(config.model_dir, 'disc_model', "data-{}_pre_embed-{}_ent-{}_exp-{}_teacher-{}".format(
                    config.data_id,
                    config.pre_embed,
                    config.ent_weight,
                    config.exp_id,
                    config.teacher_forcing)))

        ckpt = tf.train.get_checkpoint_state(disc_ckpt_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading Hier Disc model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
            # load_embeddings_disc(sess, name_scope, word2id, config.word_embedding, True)
        else:
            print("Created Hier Disc model with fresh parameters.")
            disc_global_variables = [gv for gv in tf.global_variables() if name_scope in gv.name]
            sess.run(tf.variables_initializer(disc_global_variables))
        return model

def softmax(x):
    prob = np.exp(x) / np.sum(np.exp(x), axis=0)
    return prob


def reward_evaluate(disc_config):
    beam_path = os.path.join(disc_config.beam_dir, disc_config.beam_file)
    vocab_path = os.path.join(disc_config.data_dir, "vocab%d.all" % disc_config.vocab_size)
    vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

    answer_train_ids_path = beam_path + (".ids%d.answer" % disc_config.vocab_size)
    query_train_ids_path = beam_path + (".ids%d.query" % disc_config.vocab_size)
    data_utils.data_to_token_ids(beam_path + ".gen", answer_train_ids_path, vocab)
    data_utils.data_to_token_ids(beam_path + ".query", query_train_ids_path, vocab)
    unique_list = []
    beam_set, unique_list = read_data(disc_config, query_train_ids_path, answer_train_ids_path, unique_list=unique_list)
    for set in beam_set:
        print(len(set))
    tf_config = tf.ConfigProto(allow_soft_placement=True, device_count={'GPU': 1})
    g1 = tf.Graph()
    with g1.as_default():
        sess_r = tf.Session(config=tf_config,graph=g1)
        disc_model = create_model(sess_r, disc_config, disc_config.name_model, vocab)

    buckets_num = len(disc_config.buckets)
    reward_sum = 0
    length_sum = 0
    line_num = 0
    for bucket_id in range(buckets_num):
        gen_data_reader = Gen_Data_Reader(beam_set[bucket_id])
        batch_number = gen_data_reader.get_batch_num(disc_config.batch_size)
        line_num += batch_number * disc_config.batch_size
        for batch_id in range(batch_number):
            train_batch = gen_data_reader.generate_testing_batch(disc_config.batch_size)
            train_query, train_answer = get_beam_batch(train_batch, disc_config.batch_size)
            reward_batch_sum, length_batch_sum = reward_beam_fetch(sess=sess_r, disc_config=disc_config,
                                                 disc_model=disc_model,
                                                 bucket_id=bucket_id,
                                                 queries=train_query,
                                                 answers=train_answer)
            reward_sum += reward_batch_sum
            length_sum += length_batch_sum
    print('the average reward of {}: {}'.format(disc_config.beam_file, reward_sum/line_num))
    print('the average length of {}: {}'.format(disc_config.beam_file, length_sum/line_num))


def get_beam_batch(train_data,batch_size):
    batch_source_encoder, batch_source_decoder = [], []
    for batch_i in range(batch_size):
        encoder_input, decoder_input = train_data[batch_i]
        batch_source_encoder.append(encoder_input)
        batch_source_decoder.append(decoder_input)
    return batch_source_encoder, batch_source_decoder

def reward_beam_fetch(sess, disc_config, disc_model, bucket_id, queries, answers):
    encoder_size, decoder_size = disc_config.buckets[bucket_id]
    encoder_rew, decoder_rew = [], []
    answer_len=[]
    for q_line, a_line in zip(queries, answers):
        encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(q_line))
        encoder_rew.append(q_line + encoder_pad)
        decoder_pad_size = decoder_size - len(a_line) - 1
        answer_len.append(len(a_line)+1)
        ta_line = a_line + [data_utils.EOS_ID] + [data_utils.PAD_ID] * decoder_pad_size
        decoder_rew.append(ta_line[:decoder_size])
    encoder_rew = np.array(encoder_rew)
    decoder_rew = np.array(decoder_rew)
    reward = disc_model.get_reward_forward(sess=sess,
                                           bucket_id=bucket_id,
                                           train_query_source=encoder_rew,
                                           train_answer_source=decoder_rew,
                                           train_pad_weights=None
                                           )
    reward_arr = np.array(reward).reshape(decoder_rew.shape[1], decoder_rew.shape[0])
    reward_arr = np.transpose(reward_arr)  # batch_size x decoder_size
    reward_arr = reward_arr.sum(axis=1)/np.array(answer_len)
    return reward_arr.sum(), np.sum(answer_len)

def main(_):
    args = parse_args()
    disc_config = param_create(args)
    reward_evaluate(disc_config)


if __name__ == "__main__":
    tf.app.run()
