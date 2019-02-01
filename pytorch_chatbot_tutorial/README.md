# Pytorch Chatbot Tutorial

This is the tutorial at: https://pytorch.org/tutorials/beginner/chatbot_tutorial.html

## Data

The Cornell Movie-Dialogs Corpus is a rich dataset of movie character dialog.

- 220,579 conversational exchanges between 10,292 pairs of movie characters
- 9,035 characters from 617 movies
- 304,713 total utterances

 Download the data from https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html

create a `data` folder and copy the unzipped data folder with name `cornell movie-dialogs corpus` into the new folder.

```
└── data
    ├── cornell\ movie-dialogs\ corpus
    │   ├── README.txt
    │   ├── chameleons.pdf
    │   ├── formatted_movie_lines.txt
    │   ├── movie_characters_metadata.txt
    │   ├── movie_conversations.txt
    │   ├── movie_lines.txt
    │   ├── movie_titles_metadata.txt
    │   └── raw_script_urls.txt
 ```

## Setup and Run

This tutorial requires PyTorch 1.0 or later. We recommend running the code in a virtual environment with Python 3.6:
```
virtualenv -p python3.6 venv
source venv/bin/activate
pip3 install -r requirements.txt
```
Run `jupyter notebook`

deactivate `source deactivate`

## Time to Train

Training time on a MacBook Pro with the following configurations:

```
CPU times: user 2h 2min, sys: 21min 33s, total: 2h 23min 33s
Wall time: 1h 36min 22s
```

Run `$ system_profiler SPHardwareDataType` to get MBP hardware info

```
Hardware Overview:

Model Name: MacBook Pro
Processor Name: Intel Core i5
Processor Speed: 3.1 GHz
Number of Processors: 1
Total Number of Cores: 2
Memory: 8 GB
```

## Sample Result

```
> hi
Bot: hi .
> what's your name
Bot: erica .
> what's your name
Bot: erica .
> i feel happy
Bot: you re a good man .
> where to eat?
Bot: i don t know .
> play a song for me
Bot: you know what i mean .
> where to meet
Bot: to the trunk .
> how old are you
Bot: twenty five .
> nice
Bot: you re a good man .
> what's your hobbies?
Error: Encountered unknown word.
> bummer
Error: Encountered unknown word.
> hi again
Bot: hi .
> ok, bye
Bot: bye bye .
```


## Deprecated Conda Setup

The nightly build is not available for pip yet. We have to use conda to install:

- install miniconda on Mac: https://conda.io/docs/user-guide/install/macos.html, remember to `source /Users/harrywang/.bash_profile`
We recommend running the code in conda virtual environment with Python 3.x:
```
conda create -n chatbot python=3.6
source activate chatbot
conda install pytorch-nightly -c pytorch
```
