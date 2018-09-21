# What is Rasa

This folder has code examples to play with Rasa:

Rasa has 3 parts:

1. **NLU** (Natural Language Understanding)  
This part has a customizable pipeline which includes entity recognition, feature extraction and model training.
This supervised machine learning model is to classify a given text to a predefined "intent".  https://github.com/RasaHQ/rasa_nlu

2. **CORE**   
This part is to handle the messages and assign an action of the robot. The choice of the action could be entirely hard-coded **one to one** mapping of each intent with a certain action like just print out a message. It also can be predicted by the **pretrained model** based on the chat history. https://github.com/RasaHQ/rasa_core

3. **SDK**  
This software development kit is installed in the server. If we want to customize the action, we need to install this. eg.take actions to run the query in the database and define how front-end robot interact with the customer.


# Setup

```bash
virtualenv -p python3 venv
source venv/bin/activate
pip3 install -r requirements.txt
```

# Files for training
### Description
- **stories.md**   
 - stories can be interpreted in 2 ways:
   -  a predefined script that robot should follow. we use (`memorization policy`), predict next action with probability 1 if it appears in the story line,predict with probability 0 if it never occurs.
   -  sequences of past conversation that model should be trained on (`keras Policy`) LSTM model use sequence of sentences and actions as the feature.

- **domain.yml**
 - define intents, entities, slots, actions

- **nlu.md**
 - has nlu training data(pair of sentences and intent)
- **nlu_config.yml**
 - defines the language and the embedding pipeline (tensorflow by default)

### File format
(later)

# Files generated
```
current
 ├── nlu
 │     ├── checkpoint  
 │     ├── intent_classifier_****
 │     ├── intent_featurizer_count_vectore.pkl
 │     ├── training_data.json
 │     └── metadata.json
 ├── dialogue
     ├── policy_0_FallbackPolicy
           ├── fallback_policy.json
     ├── policy_1_MemoizationPolicy
          ├── featurizer.json
          └── memorized_turns.json
     └── policy_2_keras_Policy
           ├── featurizer.json
           ├── kreas_policy.json
           └── keras.model.h5           
     ├── domain.json
     ├── domain.yml
     └── policy_metadata.json
```
### explaination
 1. **Checkpoint** it is used stored past models, as new functionality comes in we might add new data to train. check point help us manipulate among different old version

2. **intent_classifer** is related to the features and models

3. **policy** is how your robot react to the response, it has different type.
fallback_policy: when robot failed to predict your intent. memorization_policy: robot to strictly follow the storied.md
kearas policy:  use keras to train the model (you could define architecture by yourself)  

4. **domain.json** and **domain.yml** contain the same information when you defined in domain.yml in the first place  

5. **policy_metadata** store the information about evaluation of different policies, and ensemble method.


# Run
In the current folder, run the following commands

- train core dialogue: `python -m rasa_core.train -d domain.yml -s stories.md -o models/dialogue`. tensorflow is used to train models and a new folder named `models` will be created after this command.

- train NLU: `python -m rasa_nlu.train -c nlu_config.yml --data nlu.md -o models --fixed_model_name nlu --project current --verbose`， `current` folder is created under `model` after this command.
- start the conversation: `python -m rasa_core.run -d models/dialogue -u models/current/nlu`, now a web server has beens started (a flask server to serve APIs at port 5005) and you can interact with the bot using command line.
- end this conversation by typing:  `/stop`

# Functionality Component
1. **NLU**  
 - HttpInterpreter:
    - use external server to interpret user messages
 - NLUInterpreter:
    - use classification model
    - train bunches of (sentence,intent) pairs
    - features can be a combination of structured infomation or just word2vector
 - RegexInterpreter:  
     - use regular expression to handle fixed format info
     - eg. My name is Oneconnect ->/greet{"name": "Oneconnect"}

2. **Core**
 - Agent
 - Policy
 - Action
 - Statetracker
 - Slot

3. **SDK**
 - Action
    - a combination of specific task/event
    - eg. fetch info from database, change record of customer,throwback messages
 - Event
    - General Purpose Events
      - set a slot
      - restart conversation
      - schedule a reminder
    - Automated tracked Event
     - sent a message
     - undo an action

# Model Part Training
### NLU pipeline
1. Preprocess
 * X:  sentences -> Word2Vec  
 * Y: intent -> categorical variable | bags of word representation  
 Spacy_sklearn (trained model)
 tensorflow_embedding(untrained)

2. Entity extraction  
Extract entity from the received messages, the entity extracted is used for run query.

3. Classification  
keyword_Intent_Classifier  
embedding_Intent_classifier  
sklearn_Intent_classifier  

### Core pipeline
this is part use training dataset `stories.md`  
Use tensorflow to train the LSTM model based on a series of (intent,reaction)


# Potential Issue

- Intent has to be predefined and hard-coded
- If we set up two many intents, we will need more data to train the model(the curse of dimensionality)
- Multiple Intent might cause *imbalanced data* problem. It might fail to predict the category which has less training data.

To sum up, this is a good start and useful for structured Q&A
We can build on rasa and introduce
- use RNN(sequence to sequence model) to generate more natural language (rasa external language response service)
- might build a knowledge-base self-learning framework.(conference paper)
