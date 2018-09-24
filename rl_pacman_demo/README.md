# Playing Pacman: Building a Reinforcement Learning Agent

Demo on how to build a Deep Q-Network (DQN) [1] using PyTorch [2] with environment provided by OpenAI Gym [3].

## Setup

for Mac: `brew install cmake`

We recommend running the code in a virtual environment with Python > 3.5.x (fully tested on Python 3.6.5):
```
virtualenv -p python3 venv
source venv/bin/activate
pip3 install -r requirements.txt
```

Run `jupyter notebook`

## Pre-Trained Model
The following are the parameters used for the pre-trained model:
```
batch_size = 128
lr = 1e-3
betas=(0.9, 0.999)
epsilon = 0.4 # greedy exploration
gamma = 0.9
tau = 200 # num steps to update Q_target
```
Note that both the `q_network` and `q_target` have two fc-layers of size `layer_sizes=[256,256]`.

## References

1. http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12389/11847
2. https://github.com/pytorch/pytorch
3. https://github.com/openai/gym
