# Train models on the DSTC7 task 2 reddit dataset using ParlAI framework

ParlAI (pronounced “par-lay”) is a framework for dialog AI **research**, implemented in Python

## Step 1: Install [a customized ParlAI](https://github.com/outformatics/ParlAI/tree/parlai4reddit) with built-in reddit data support

```bash
cd YOUR_PATH_TO_BUFFET/buffet/ParlAI
virtualenv venv -p python3.6
source venv/bin/activate
pip install -r requirements.txt
```

## Step 2: get the DSTC7 reddit dataset into ParlAI framework

Create the data directory YOUR_BUFFET_PATH/ParlAI/venv/lib/python3.6/site-packages/data/reddit

```bash
mkdir -p YOUR_BUFFET_PATH/ParlAI/venv/lib/python3.6/site-packages/data/reddit
```

Put the reddit dataset into the directory YOUR_BUFFET_PATH/ParlAI/venv/lib/python3.6/site-packages/data/reddit

Make sure to change the name of the dataset into **train.txt**, **valid.txt**, and **test.txt** 

[Optional] More about the built-in datasets in ParlAI: 
- The built-in datasets will be downloaded to YOUR_BUFFET_PATH/ParlAI/venv/lib/python3.6/site-packages/data when running the [examples](https://github.com/gamma-lab/buffet/blob/master/ParlAI/examples). 

## Step 3: Display the reddit data

```python
python examples/display_data.py -t reddit
```

## Step 4: Trained a seq2seq baseline model by running

```python
python examples/train_reddit.py
```

