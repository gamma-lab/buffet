import random
# from al_neural_dialogue_train import disc_train_data
import numpy as np
from six.moves import xrange
from utils import data_utils
import heapq


class Gen_Data_Reader:
    def __init__(self, train_set=None):
        self.train_index = 0
        self.train_data = train_set
        self.data_size = len(train_set)
        self.shuffle_index_list = self.shuffle_index()
        self.original_index = list(range(self.data_size))

    def get_batch_num(self, batch_size):
        # in each batch, half of the samples are from human while another half from machine.
        x_batch = self.data_size % batch_size
        if x_batch == 0:
            return int(self.data_size / batch_size)
        else:
            return int(self.data_size / batch_size + 1)

    def shuffle_index(self):
        shuffle_index_list = random.sample(range(self.data_size), self.data_size)
        return shuffle_index_list

    def generate_batch_index(self, batch_size):
        if self.train_index + batch_size > self.data_size:
            batch_index = self.shuffle_index_list[self.train_index:self.data_size]
            self.shuffle_index_list = self.shuffle_index()
            remain_size = batch_size - (self.data_size - self.train_index)
            batch_index += self.shuffle_index_list[:remain_size]
            self.train_index = remain_size
        else:
            batch_index = self.shuffle_index_list[self.train_index:self.train_index + batch_size]
            self.train_index += batch_size
        return batch_index

    def generate_testing_batch_index(self, batch_size):
        if self.train_index + batch_size > self.data_size:
            batch_index = self.original_index[self.train_index:self.data_size]
            remain_size = batch_size - (self.data_size - self.train_index)
            batch_index += self.original_index[:remain_size]
            self.train_index = remain_size
        else:
            batch_index = self.original_index[self.train_index:self.train_index + batch_size]
            self.train_index += batch_size
        return batch_index

    def generate_testing_batch(self, batch_size):
        test_set = []
        test_index = self.generate_testing_batch_index(batch_size)
        for index in test_index:
            test_set.append(self.train_data[index])
        return test_set

    def generate_training_batch(self, batch_size):
        train_set = []
        training_index = self.generate_batch_index(batch_size)
        for index in training_index:
            train_set.append(self.train_data[index])
        return train_set


def eval_step(sess=None, bucket_id=None, disc_model=None, train_query=None, train_answer=None, train_labels=None,
              forward_only=False):
    feed_dict = {}

    for i in xrange(len(train_query)):
        feed_dict[disc_model.query[i].name] = train_query[i]

    for i in xrange(len(train_answer)):
        feed_dict[disc_model.answer[i].name] = train_answer[i]

    feed_dict[disc_model.target.name] = train_labels

    loss = 0.0
    if forward_only:
        fetches = [disc_model.b_logits_1[bucket_id]]
        logits = sess.run(fetches, feed_dict)
        logits = logits[0]
    else:
        fetches = [disc_model.b_train_op[bucket_id], disc_model.b_loss[bucket_id], disc_model.b_logits_1[bucket_id]]
        train_op, loss, logits = sess.run(fetches, feed_dict)

    # softmax operation
    logits = np.transpose(softmax(np.transpose(logits)))

    reward_all, gen_num = 0.0, 0
    correct_num = 0
    for logit, label in zip(logits, train_labels):
        if int(label) == 0:
            reward_all += logit[1]
            gen_num += 1
        if logit[int(label)] > logit[1 - int(label)]:
            correct_num += 1
    reward = reward_all / gen_num

    return reward, correct_num * 1.0 / len(train_labels), [reward_all, gen_num, correct_num, len(train_labels)]


def reward_eval_step():
    None


def beam_decoding(config, sess, gen_model, source_inputs, source_outputs,
                  encoder_inputs, decoder_inputs, target_weights, target_inputs, bucket_id, mc_search=False):
    answer_len = config.buckets[bucket_id][1]
    beam_size = config.beam_size
    batch_size = config.batch_size
    beam_encoder_inputs = []
    encoder_inputs_list = np.transpose(encoder_inputs).tolist()
    for each_encoder_input in encoder_inputs_list:
        for _ in range(config.beam_size):
            beam_encoder_inputs.append(each_encoder_input)
    beam_encoder_inputs = np.transpose(beam_encoder_inputs)

    def single_example(prob):
        prob = prob.ravel()
        word_index_ = heapq.nlargest(beam_size, range(len(prob)), prob.take)
        word_prob_ = prob[word_index_].tolist()
        return word_index_, word_prob_

    def same_prefix_search(prefix_word, prefix_prob, prob_same_prefix):
        # len(prob_same_prefix) = beam_size
        # all lines have the same encoder_input
        # print(len(prefix_word), len(prefix_prob), len(prob_same_prefix))
        word_index, word_prob = [], []
        largest_index, largest_prob = [], []
        for idx, beam_each_line in enumerate(prob_same_prefix):
            if prefix_word[idx][-1] == data_utils.EOS_ID:
                for _ in range(beam_size):
                    word_index.append(prefix_word[idx] + [data_utils.EOS_ID])
                    word_prob.append(prefix_prob[idx] * 1.0)
            else:
                word_index_single, word_prob_single = single_example(beam_each_line)
                for each_word, each_word_prob in zip(word_index_single, word_prob_single):
                    # print(prefix_prob[idx], each_word_prob, each_word)
                    word_index.append(prefix_word[idx] + [each_word])
                    word_prob.append(prefix_prob[idx] * each_word_prob)
        word_prob_array = np.array(word_prob)
        word_prob_index = np.argsort(-word_prob_array)
        ix = 0
        while len(largest_index) < beam_size:
            index = word_prob_index[ix]
            if word_index[index] not in largest_index:
                largest_index.append(word_index[index])
                largest_prob.append(word_prob[index])
            ix += 1
        # largest_index_top = heapq.nlargest(beam_size, range(len(word_prob_array)), word_prob_array.take)
        # for idx in largest_index_top:
        #     largest_index.append(word_index[idx])
        #     largest_prob.append(word_prob[idx])
        # print(np.array(largest_index).shape, np.array(largest_prob).shape)
        return largest_index, largest_prob

    def find_beam_results(word_index, word_prob):
        beam_results = []
        for batch_index in range(0, len(word_index), beam_size):
            beam_word = word_index[batch_index:batch_index + beam_size]
            beam_prob = word_prob[batch_index:batch_index + beam_size]
            max_index = np.argmax(beam_prob)
            max_pos = 2
            if beam_word[max_index][1] == data_utils.EOS_ID and max_pos <= len(beam_prob):
                max_index = heapq.nlargest(max_pos, xrange(len(beam_prob)), key=beam_prob.__getitem__)[max_pos - 1]
                max_pos += 1
            beam_results.append(beam_word[max_index])
        return beam_results

    def fill_pad_pos(beam_input, dec_len):
        # beam_input: beam_size * batch_size X decoder_len
        # print(len(beam_input), len(beam_input[0]))
        if len(beam_input[0]) == dec_len:
            return beam_input
        elif len(beam_input[0]) > dec_len:
            raise ValueError("beam input length should be smaller than decoder size")
        else:
            padded_result = []
            padded_part = [data_utils.PAD_ID] * (dec_len - len(beam_input[0]))
            for line in beam_input:
                padded_result.append(line + padded_part)
            return padded_result

    # first position
    word_index = [[data_utils.GO_ID]] * beam_size * batch_size
    # beam_size * batch_size X decoder_len
    word_prob = [1.0] * beam_size * batch_size

    for dec_position in range(answer_len):
        padded_word_index = fill_pad_pos(word_index, answer_len)
        _, _, _, output_logits = gen_model.step(sess,
                                                beam_encoder_inputs,
                                                np.transpose(padded_word_index),
                                                target_inputs,
                                                target_weights,
                                                bucket_id=bucket_id,
                                                reward=1,
                                                forward_only=True,
                                                mc_search=mc_search,
                                                mc_position=100)
        logit_last = np.transpose(softmax(np.transpose(output_logits[dec_position])))
        word_index_tmp, word_prob_tmp = [], []
        for batch_index in range(0, len(logit_last), beam_size):
            same_prefix_in_batch_prob = logit_last[batch_index:batch_index + beam_size]
            same_prefix_input, same_prefix_prob = same_prefix_search(
                prefix_word=word_index[batch_index:batch_index + beam_size],
                prefix_prob=word_prob[batch_index:batch_index + beam_size],
                prob_same_prefix=same_prefix_in_batch_prob)
            for beam_word, beam_prob in zip(same_prefix_input, same_prefix_prob):
                word_index_tmp.append(beam_word)
                word_prob_tmp.append(beam_prob)
        word_index = word_index_tmp
        word_prob = word_prob_tmp

    beam_results = find_beam_results(word_index, word_prob)
    resps = []
    for seq in beam_results:
        if data_utils.EOS_ID in seq:
            resps.append(seq[1:seq.index(data_utils.EOS_ID)][:config.buckets[bucket_id][1]])
        else:
            resps.append(seq[1:config.buckets[bucket_id][1]])
    query_list = source_inputs + source_inputs
    answer_list = source_outputs + resps
    label_list = [1] * len(source_outputs) + [0] * len(resps)

    return query_list, answer_list, label_list, []


def beam_decoding_with_reward(config, sess, gen_model, reward_model, source_inputs, source_outputs,
                              encoder_inputs, decoder_inputs, target_weights, target_inputs, bucket_id,
                              mc_search=False):
    answer_len = config.buckets[bucket_id][1]
    beam_size = config.beam_size
    batch_size = config.batch_size
    beam_encoder_inputs = []
    encoder_inputs_list = np.transpose(encoder_inputs).tolist()
    for each_encoder_input in encoder_inputs_list:
        for _ in range(config.beam_size):
            beam_encoder_inputs.append(each_encoder_input)
    beam_encoder_inputs = np.transpose(beam_encoder_inputs)

    def single_example(prob):
        # find the top-k high prob lines
        prob = prob.ravel()
        word_index_ = heapq.nlargest(beam_size, range(len(prob)), prob.take)
        word_prob_ = prob[word_index_].tolist()
        return word_index_, word_prob_

    def reward_fetch(queries, answers):
        encoder_size, decoder_size = config.buckets[bucket_id]
        encoder_rew, decoder_rew = [], []
        for q_line, a_line in zip(queries, answers):
            encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(q_line))
            encoder_rew.append(q_line + encoder_pad)
            # remove 'GO_ID' 
            if a_line[0] == data_utils.GO_ID:
                a_line = a_line[1:]
            if data_utils.EOS_ID in a_line:
                a_line = a_line[:a_line.index(data_utils.EOS_ID)+1]

            decoder_pad_size = decoder_size - len(a_line)
            ta_line = a_line + [data_utils.PAD_ID] * decoder_pad_size
            decoder_rew.append(ta_line[:decoder_size])
        encoder_rew = np.array(encoder_rew)
        decoder_rew = np.array(decoder_rew)
        reward = reward_model.get_reward_forward(sess=sess[1],
                                                 bucket_id=bucket_id,
                                                 train_query_source=encoder_rew,
                                                 train_answer_source=decoder_rew,
                                                 train_pad_weights=None
                                                 )
        reward_arr = np.array(reward).reshape(decoder_rew.shape[1], decoder_rew.shape[0])
        reward_arr = np.transpose(reward_arr)  # batch_size x decoder_size
        return reward_arr

    def find_high_reward_sample(word_index_single, prefix_word, idx, source_encoder_input_single):
        answer_input = []
        query_input = []
        word_index_max_list = []
        reward_max_list = []
        # store the top-k samples that will be fed to reward model
        for each_word in word_index_single:
            answer_input.append(prefix_word[idx] + [each_word])
            query_input.append(source_encoder_input_single)
        reward_group = reward_fetch(query_input, answer_input)
        reward_traj = reward_group.sum(axis=1)
        reward_list = reward_traj.ravel()
        word_max = heapq.nlargest(beam_size, range(len(reward_list)), reward_list.take)
        for id_word in word_max:
            word_index_max_list.append(word_index_single[id_word])
            reward_max_list.append(reward_list[id_word])
        return word_index_max_list, reward_max_list

    def same_prefix_search(prefix_word, prefix_reward, prob_same_prefix, source_encoder_input_bline):
        # len(prob_same_prefix) = beam_size
        # all lines have the same encoder_input
        # print(len(prefix_word), len(prefix_prob), len(prob_same_prefix))
        word_index_inside, word_reward_inside = [], []
        largest_index, largest_reward = [], []
        for idx, beam_each_line in enumerate(prob_same_prefix):
            if prefix_word[idx][-1] == data_utils.EOS_ID:
                for line_num in range(beam_size):
                    word_index_inside.append(prefix_word[idx] + [data_utils.EOS_ID])
                    # if last token is EOS, keep only one copy and discard others
                    if line_num == 0:
                        word_reward_inside.append(prefix_reward[idx])
                    else:
                        word_reward_inside.append(0)
            else:
                word_index_single, word_prob_single = single_example(beam_each_line)
                word_index_single, word_reward_single = find_high_reward_sample(word_index_single=word_index_single,
                                                                                prefix_word=prefix_word,
                                                                                idx=idx,
                                                                                source_encoder_input_single=source_encoder_input_bline)
                for each_word, each_reward in zip(word_index_single, word_reward_single):
                    word_index_inside.append(prefix_word[idx] + [each_word])
                    word_reward_inside.append(each_reward)

        word_reward_array = np.array(word_reward_inside)
        word_reward_index = np.argsort(-word_reward_array)
        ix = 0
        while len(largest_index) < beam_size:
            index = word_reward_index[ix]
            if word_index_inside[index] not in largest_index:
                largest_index.append(word_index_inside[index])
                largest_reward.append(word_reward_inside[index])
            ix += 1
        return largest_index, largest_reward

    def find_beam_results(word_index, word_reward):
        beam_results = []
        for batch_index in range(0, len(word_index), beam_size):
            beam_word = word_index[batch_index:batch_index + beam_size]
            beam_reward = word_reward[batch_index:batch_index + beam_size]
            max_index = np.argmax(beam_reward)
            max_pos = 2
            if beam_word[max_index][1] == data_utils.EOS_ID and max_pos <= len(beam_reward):
                max_index = heapq.nlargest(max_pos, xrange(len(beam_reward)), key=beam_reward.__getitem__)[max_pos - 1]
                max_pos += 1
            beam_results.append(beam_word[max_index])
        return beam_results

    def fill_pad_pos(beam_input, dec_len):
        # beam_input: beam_size * batch_size X decoder_len
        # print(len(beam_input), len(beam_input[0]))
        if len(beam_input[0]) == dec_len:
            return beam_input
        elif len(beam_input[0]) > dec_len:
            raise ValueError("beam input length should be smaller than decoder size")
        else:
            padded_result = []
            padded_part = [data_utils.PAD_ID] * (dec_len - len(beam_input[0]))
            for line in beam_input:
                padded_result.append(line + padded_part)
            return padded_result

    # first position
    word_index = [[data_utils.GO_ID]] * beam_size * batch_size
    # beam_size * batch_size X decoder_len
    word_reward = [0] * beam_size * batch_size

    for dec_position in range(answer_len):
        padded_word_index = fill_pad_pos(word_index, answer_len)
        _, _, _, output_logits = gen_model.step(sess[0],
                                                beam_encoder_inputs,
                                                np.transpose(padded_word_index),
                                                target_inputs,
                                                target_weights,
                                                bucket_id=bucket_id,
                                                reward=1,
                                                forward_only=True,
                                                mc_search=mc_search,
                                                mc_position=100)
        logit_last = np.transpose(softmax(np.transpose(output_logits[dec_position])))
        word_index_tmp, word_reward_tmp = [], []
        for batch_index in range(0, len(logit_last), beam_size):
            same_prefix_in_batch_prob = logit_last[batch_index:batch_index + beam_size]
            same_prefix_input, same_prefix_reward = same_prefix_search(
                prefix_word=word_index[batch_index:batch_index + beam_size],
                prefix_reward=word_reward[batch_index:batch_index + beam_size],
                prob_same_prefix=same_prefix_in_batch_prob,
                source_encoder_input_bline=source_inputs[int(batch_index / beam_size)])
            for beam_word, beam_reward in zip(same_prefix_input, same_prefix_reward):
                word_index_tmp.append(beam_word)
                word_reward_tmp.append(beam_reward)
        word_index = word_index_tmp
        word_reward = word_reward_tmp

    beam_results = find_beam_results(word_index, word_reward)
    resps = []
    for seq in beam_results:
        if data_utils.EOS_ID in seq:
            resps.append(seq[1:seq.index(data_utils.EOS_ID)][:config.buckets[bucket_id][1]])
        else:
            resps.append(seq[1:config.buckets[bucket_id][1]])
    query_list = source_inputs + source_inputs
    answer_list = source_outputs + resps
    label_list = [1] * len(source_outputs) + [0] * len(resps)

    return query_list, answer_list, label_list, []


def softmax(y):
    x = y.copy()
    x_max = np.max(x, axis=0)
    for i in range(len(x)):
        x[i] -= x_max
    prob = np.exp(x) / np.sum(np.exp(x), axis=0)
    return prob
