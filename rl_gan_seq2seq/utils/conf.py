import os

all_ent_weight = 0.1
vocab_size = 10000
#vocab_size = 20000
word_embedding_size = 200
num_roll = 1
pre_embed_flag = False
model_dir_all = './model'
data_id = 'dailydialog'
adv = True
teacher_forcing = True
buckets = [(20, 10), (40, 20), (60, 30), (81, 30)]
continue_train = True
exp_id = 10
testing_flag = False
num_step_multi = 2

# exp==4-> 4 layers GRU
# configuration options for discriminator network
class disc_config(object):
    data_dir = os.path.join('./data', data_id, 'subdata')
    model_dir = model_dir_all
    data_id = data_id
    exp_id = exp_id
    adv = adv
    batch_size = 64
    ent_weight = all_ent_weight
    beam_dir=''
    beam_file = ''
    learning_rate = 0.001
    continue_train = continue_train
    learning_rate_decay_factor = 0.99995
    vocab_size = vocab_size
    embed_dim = 512
    word_embedding = word_embedding_size
    pre_embed = pre_embed_flag
    steps_per_checkpoint = 30*num_step_multi
    #hidden_neural_size = 128
    num_layers = 2
    name_model = "disc_model"
    tensorboard_dir = "./tensorboard/disc_log/"
    name_loss = "disc_loss"
    max_len = 50
    piece_size = batch_size * steps_per_checkpoint
    piece_dir = "./disc_data/batch_piece/"
    valid_num = 100
    init_scale = 0.1
    num_class = 2
    keep_prob = 0.80
    train_ratio=0.5
    max_grad_norm = 5
    buckets = buckets
    teacher_forcing = teacher_forcing



# configuration options for generator network
class gen_config(object):
    exp_id = exp_id
    teacher_forcing = teacher_forcing
    data_dir = os.path.join('../dialogue-gan/data', data_id)
    data_id = data_id
    model_dir = model_dir_all
    continue_train = continue_train
    adv = adv
    testing = testing_flag
    num_roll = num_roll
    beam_size = 8
    repeat_word = 0.1
    learning_rate = 0.001
    learning_rate_decay_factor = 0.99999
    max_gradient_norm = 5.0
    batch_size = 64
    ent_weight = all_ent_weight
    emb_dim = 512
    word_embedding = word_embedding_size
    pre_embed = pre_embed_flag
    num_layers = 2
    vocab_size = vocab_size
    name_model = "st_model"
    tensorboard_dir = "./tensorboard/gen_log/"
    name_loss = "gen_loss"
    teacher_loss = "teacher_loss"
    reward_name = "reward"
    max_train_data_size = 0
    steps_per_checkpoint = 20*num_step_multi
    buckets = buckets
    keep_prob = 0.8



class GSTConfig(object):
    beam_size = 7
    learning_rate = 0.5
    learning_rate_decay_factor = 0.99
    max_gradient_norm = 5.0
    batch_size = 256
    emb_dim = 1024
    num_layers = 2
    vocab_size = 7200
    train_dir = "./gst_data/"
    name_model = "st_model"
    tensorboard_dir = "./tensorboard/gst_log/"
    name_loss = "gst_loss"
    max_train_data_size = 0
    steps_per_checkpoint = 200*num_step_multi
    buckets =        [(5, 10), (10, 15), (20, 25), (40, 50)]
    buckets_concat = [(5, 10), (10, 15), (20, 25), (40, 50), (100, 50)]
