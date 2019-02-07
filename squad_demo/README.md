# SQuAD Reading Comprehension


# Setup Instructions

1. Clone this repository
2. Download and Extract https://s3.amazonaws.com/lily-models/bert_pytorch_model.bin.zip
3. Setup virtual env with Python 3.x `virtualenv venv` -> `source venv/bin/activate`
4. `pip install -r requirements.txt`


# Python Interface

_The script is configured to use CPU in `squad._load()` function. Executing a batch is significantly fast on GPU._

```python
import squad
print(squad.qa_system_predict(paragraph_text, question_text))
```

# HTTP Interface

1. `python main.py`
2. Open http://localhost:8881
