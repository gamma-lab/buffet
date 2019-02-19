This repository is for the code of the AAAI-2019 paper:

[**Dialogue Generation: From Imitation Learning to Inverse Reinforcement Learning**](https://arxiv.org/abs/1812.03509)

## Set-up & Config:
GPU support
TensorFlow 1.10.0  Python 3.6

## Reinforcement & GAN
It brilliantly leverage discriminator for a reward mechanism for Reinforcement Learning.
The Generator pick next word based on previous seen words and will recieve a reward from the discriminator. Another way to do this is using Inverse Reinforcement Learning, which builds a deep neural network to learn the reward function based on the training data. The underlying intuition is that the training data is generated under the guidance of unknown reward function.

## Dataset:
Dailydialog: the original dataset of daily chichat
GoalDialog: the dataset I found with goal oriented dialogue.
All the dataset conatins one-turn conversation. Both training and testing are (Q,A) pairs.
The context-reply pairs should be saved in two different files. e.g. train.query, train.answer, dev.query, dev.answer

## Run
you could simply utilize makefile
```
make train
```
or specifically run in the command line when you are in the main folder
```
python -u irl-gan.py -data_id dailydialog -vocab_size 0 -hidden_size 1024 -ent_weight 0.005 -exp_id 2 --adv_train --teacher_forcing --no_testing --no_continue_train
```
**parameter**  
data_id: training dataset folder  
vocab_size:  the threshold of word frequency  
hidden_size: GRU hidden size  
ent_weight: penalizing weight for entropy, robustness and randomness trade off  
exp_id:   exp id
adv_train:  adversarial training
teacher_forcing:  enforce teacher training for generator
no_testing/testing:  testing and beam search
no_continue_train/continue_trian: continue training for models

You can preset some parameters in file utils/conf.py eg. set the steps_per_checkpoint



**Note:** This is the code based on [Yu's code](https://github.com/LantaoYu/SeqGAN/blob/master/README.md) and [Liu's code](https://github.com/liuyuemaicha/Adversarial-Learning-for-Neural-Dialogue-Generation-in-Tensorflow). Much appreciate.
