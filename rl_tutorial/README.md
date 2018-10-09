# Reinforcement Learning with Deep Q-Network (DQN) Tutorial
This tutorial shows how we can implement a DQN with Pytorch [2] and an action replay buffer to solve a classic control problem CartPole-v0. The environment is provided by OpenAI's gym [3].  The following notebook tutorial can be found [here](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html).

## Setup
for Mac: `brew install cmake` 
We recommend running the code in a virtual environment with Python > 3.5.x (fully tested on Python 3.6.5):
```
virtualenv -p python3 venv
source venv/bin/activate
pip3 install -r requirements.txt
```

Run `jupyter notebook`

## Task
The agent's task is to balance a pole on a cart for as long as possible; hence, it has two actions: to go left or to go right. The task is solved if the agent reaches a reward of 200. More information about this environment can be found [here](https://gym.openai.com/envs/CartPole-v0/).
 
## References

1. http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12389/11847
2. https://github.com/pytorch/pytorch
3. https://github.com/openai/gym
