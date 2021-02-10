# Vincent LIU - Technical test for Descartes Underwriting Data Scientist Position

## Requirements
```
pip install - r requirements.txt
```

## Workflow

This is a binary classification problem with unbalanced dataset. We use F1-score to assess the performance of our approach. (Accuracy here is not a good metric since the dataset is unbalanced).

Our methodology consists of:

* Conducting preliminary analysis through 1-Preliminary.ipynb
* Conducting exploratory analysis on 2-EDA.ipynb
* Test simple models, for example on 4-XGBModel.ipynb
* Automate the workflow with python scripts (run.sh script for example)

As further work, we suggest performing more feature engineering to help increase the F1 score.
## Run directly
```
chmod +x run.sh
./run.sh
```

## Run manually

### Preprocess data
We split the dataset into two parts:
1. The first part is used for hyperparameters search, where we will perform k-fold.
2. The second part is used for future validation, to evaluate the model generalization ability.
```
python3 preprocess_data.py
python3 split_data.py --random_state 4
```

### Training models
```
python3 train.py --model XGB --random_state 4
```

### Evaluate ensemble models on test set
```
python3 evaluate_on_future_validation.py
```

### Generate submissions
```
python3 generate_sub.py
```
