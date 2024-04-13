This repository contains a very simple python program that taken in Twitter training and validation csv files from Kaggle and performs Sentiment analysis to classify the statement to 'Positive', 'Negative' or 'Neutral'.

In order to train the model, run eval.py:
```
python eval.py
```

The model gets saved in daved_models folder and if run again, it picks the latest from this script and runs it for the next 10 epochs.

In order to run the classification in streamlit, run:
```
streamlit run stapi.py
'''
