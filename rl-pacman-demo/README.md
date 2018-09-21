# Playing Pacman: Building a Reinforcement Learning Agent

Demo on how to build a Deep Q-Network (DQN) [1] using [PyTorch] https://github.com/pytorch/pytorch with environment provided by [OpenAI Gym] https://github.com/openai/gym

## Environment Setup
We recommend running the code in a virtual environment with Python 3.5.x:
```
virtualenv -p python3 venv
source venv/bin/activate
pip3 install -r requirements.txt
```
In addition, we need to set up the kernel for Jupyter Notebook to avoid issues with locating `gym` module:
```
python3 -m ipykernel install --user --name <envname>
```

Run `$ jupyter notebook` and select the `<envname>` kernel to run the demo.

## References
* [1] http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12389/11847
