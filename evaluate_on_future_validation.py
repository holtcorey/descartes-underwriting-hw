import numpy as np
import pandas as pd
import argparse
import os

from sklearn.metrics import f1_score
from joblib import load


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default='processed_data/train_future_validation.csv', type=str)
    parser.add_argument("--random_state", default=4, type=int)
    parser.add_argument('--model_path', default='models', type=str)
    args = parser.parse_args()

    # Load data
    X_future_validation = pd.read_csv(args.src)
    y_future_validation = X_future_validation['TARGET_FLAG'].values
    X_future_validation = X_future_validation.drop(columns=['TARGET_FLAG'])

    # Load models
    assert(os.path.exists(args.model_path))
    models_scikit = []
    if os.path.isdir(args.model_path):
        for model_path in os.listdir(args.model_path):
            if model_path.split('_')[0] in ['RF', 'XGB']:
                models_scikit.append(load(os.path.join(args.model_path, model_path)))
            else:
                pass
                # raise NotImplementedError("To be implemented")
    else:
        raise NotImplementedError("To be implemented")

    # Average output probabilities
    y_preds = []
    for model in models_scikit:
        y_pred = model.predict_proba(X_future_validation)[:, 1]
        y_preds.append(y_pred)

    y_preds = np.mean(y_preds, axis=0)
    y_preds = (y_preds > 0.5).astype(int)

    # Evaluate on never seen test set
    f1 = f1_score(y_future_validation, y_preds)
    print('f1 score on never seen test set : {}'.format(f1))