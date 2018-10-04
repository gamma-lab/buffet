#!/usr/bin/env python3

# Adapted from the scripts at ParlAI by Xin Chen

from parlai.scripts.train_model import setup_args, TrainLoop

if __name__ == '__main__':
    parser = setup_args()
    parser.set_defaults(
        task='reddit',
        model='seq2seq',
        model_file='/tmp/xchen/test_model',
        dict_file='data/reddit/temp_dict',
        gpu=0,
        dict_lower=True,
        datatype='train',
        batchsize=10,
        hiddensize=1024,
        embeddingsize=300,
        attention='none',
        numlayers=3,
        rnn_class='lstm',
        learningrate=1,
        dropout=0.1,
        gradient_clip=0.1,
        lookuptable='enc_dec',
        optimizer='sgd',
        # embedding_type='glove',
        momentum=0.9,
        bidirectional=False,
        batch_sort=False,
        validation_every_n_secs=53200,
        validation_metric='ppl',
        validation_metric_mode='min',
        validation_patience=15,
        log_every_n_secs=1,
        numsoftmax=1,
        truncate=150,
    )
    TrainLoop(parser).train()
