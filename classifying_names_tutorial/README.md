# CLASSIFYING NAMES WITH A CHARACTER-LEVEL RNN

This is the tutorial at: https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

**Author**: Sean Robertson https://github.com/spro/practical-pytorch

## Data
Download the data from
https://download.pytorch.org/tutorial/data.zip
and extract it to the current directory.

Included in the ``data/names`` directory are 18 text files named as
"[Language].txt". Each file contains a bunch of names, one name per
line, mostly romanized (but we still need to convert from Unicode to
ASCII).

We'll end up with a dictionary of lists of names per language,
``{language: [names ...]}``. The generic variables "category" and "line"
(for language and name in our case) are used for later extensibility.

```
./data/
├── eng-fra.txt
└── names
    ├── Arabic.txt
    ├── Chinese.txt
    ├── Czech.txt
    ├── Dutch.txt
    ├── English.txt
    ├── French.txt
    ├── German.txt
    ├── Greek.txt
    ├── Irish.txt
    ├── Italian.txt
    ├── Japanese.txt
    ├── Korean.txt
    ├── Polish.txt
    ├── Portuguese.txt
    ├── Russian.txt
    ├── Scottish.txt
    ├── Spanish.txt
    └── Vietnamese.txt
```

## Setup

We recommend running the code in a virtual environment with Python > 3.5.x (fully tested on Python 3.6.5):
```
virtualenv -p python3 venv
source venv/bin/activate
pip3 install -r requirements.txt
```

Run `jupyter notebook`

## Run

The entire notebook takes a few minutes (1m 20s for training process) to run on a MacBook Pro with the following configurations:
```
Hardware Overview:

      Model Name: MacBook Pro
      Processor Name: Intel Core i5
      Processor Speed: 2.3 GHz
      Number of Processors: 1
      Total Number of Cores: 4
      Memory: 16 GB
```
