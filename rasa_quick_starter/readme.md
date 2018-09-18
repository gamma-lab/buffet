# What is Rasa

This folder has code examples to play with Rasa:

Rasa has two parts:

1. **NLU** (natural luanguage understanding)
this part has a customizable pipleline which include entity recognition, feature extraction and training the models
This supervised machine learning model is to classify a given text to a predefined "intent".  https://github.com/RasaHQ/rasa_nlu

2. **CORE**   
This part is hard-coded **one to one** mapping of each intent with a certain action like function call or just print our a message.
You could also customerize a conversation flow with predifined structure. The robot will follow the "script" you hard-coded or send out some certain messages. https://github.com/RasaHQ/rasa_core


# Setup

```bash
virtualenv -p python3 venv
source venv/bin/activate
pip3 install -r requirements.txt
```

# Files

- stories.md file
- domain.yml has intents and actions
- nlu.md has nlu training data
- nlu_config.yml defines the language and the embeding pipeline (tensorflow by default)

# Run

In the current folder, run the following commands

- train core dialogue: `python -m rasa_core.train -d domain.yml -s stories.md -o models/dialogue`. tensorflow is used to train models and a new folder named `models` will be created after this command.
- train NLU: `python -m rasa_nlu.train -c nlu_config.yml --data nlu.md -o models --fixed_model_name nlu --project current --verbose`， `current` folder is created under `model` after this command.
- start the conversation: `python -m rasa_core.run -d models/dialogue -u models/current/nlu`, now a web server has beens started (a flask server to serve APIs at port 5005) and you can interact with the bot using command line.
- end this conversation by typing:  `/stop`

# Potential Issue

- Intent has to be predifined and hard coded
- Conversation structure is fixed and set-up
- If we set up two many Intents, we will need more data to train the model(curse of dimensionality)

To sum up, this is a good start and useful for structured Q&A
We believe the next generation of the robot might
- use RNN(sequence to sequence model) to generate more natural language
- might build a knowledge-base self-learning framework.
