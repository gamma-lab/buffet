# Train models on the DSTC7 task 2 reddit dataset using ParlAI framework

ParlAI (pronounced “par-lay”) is a framework for dialog AI **research**, implemented in Python

## Step 1: clone ParlAI 

Run the following commands to clone the repository:

```bash
cd YOUR_PATH_TO_BUFFET/buffet
git clone https://github.com/facebookresearch/ParlAI.git ./ParlAI
```

After cloning, you may want to delete the .git folder.

```bash
rm -fr .git
```

## Step 2: install ParlAI 

Run the following commands to install ParlAI:

```bash
cd ./ParlAI; python setup.py develop
```

This will link the cloned directory to your site-packages.

This is the recommended installation procedure, as it provides ready access to the examples and allows you to modify anything you might need. This is especially useful if you want to submit another fixed dataset or dynamic task to the repository.

Note: the step 1 and step 2 are following [the original ParlAI instructions](https://github.com/facebookresearch/ParlAI#installing-parlai) 

## Step 3: get the DSTC7 reddit dataset into ParlAI framework

Create a new directory for the reddit dataset 

```bash
mkdir -p ./parlai/tasks/reddit
```
Create [three python files](https://github.com/gamma-lab/buffet/blob/master/ParlAI/parlai/agents/reddit) into the new directory. 

Create the data directory YOUR_BUFFET_PATH/ParlAI/data/reddit

```bash
mkdir -p YOUR_BUFFET_PATH/ParlAI/data/reddit
```

Put the reddit dataset into the directory YOUR_BUFFET_PATH/ParlAI/data/reddit

Make sure to change the name of the dataset into **train.txt**, **valid.txt**, and **test.txt** 

Display the reddit data

```python
python examples/display_data.py -t reddit
```

[Optional] More about the built-in datasets in ParlAI: 
- The built-in datasets will be downloaded to YOUR_BUFFET_PATH/ParlAI/data when running the [examples](https://github.com/gamma-lab/buffet/blob/master/ParlAI/examples). 
- Any non-data files (such as the MemNN code) if requested will be downloaded to YOUR_BUFFET_PATH/ParlAI/downloads. 
- If you need to clear out the space used by these files, you can safely delete these directories and any files needed will be downloaded again.

Please refer to the doc on [Getting a New Dataset Into ParlAI](http://parl.ai/static/docs/tutorial_task.html#creating-a-new-task-the-more-complete-way) for more detail.

## Step 4: Create a directory for your research project based on ParlAI

```bash
mkdir -p YOUR_BUFFET_PATH/ParlAI/projects/reddit
cd YOUR_BUFFET_PATH/ParlAI
```

## Step 5: Create [a python script](https://github.com/gamma-lab/buffet/blob/master/ParlAI/projects/reddit/train_reddit.py) that trained a seq2seq baseline model by running

```python
python ./projects/reddit/train_reddit.py
```

