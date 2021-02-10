import numpy as np
import pandas as pd
import argparse
import os

from joblib import load

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default='processed_data/test.csv', type=str)
    parser.add_argument("--dst", default='submissions/submission.csv', type=str)
    parser.add_argument('--model_path', default='models', type=str)
    args = parser.parse_args()

    # Load data
    X_test = pd.read_csv(args.src)
    X_test.drop(columns=['TARGET_FLAG'], inplace=True)

    # Load models
    assert(os.path.exists(args.model_path))
    models_scikit = []
    if os.path.isdir(args.model_path):
        for model_path in os.listdir(args.model_path):
            if model_path.split('_')[0] in ['RF', 'XGB']:
                models_scikit.append(load(os.path.join(args.model_path, model_path)))
            else:
                pass
    else:
        raise NotImplementedError("To be implemented")

    # Average output probabilities
    y_preds = []
    for model in models_scikit:
        y_pred = model.predict_proba(X_test)[:, 1]
        y_preds.append(y_pred)

    y_preds = np.mean(y_preds, axis=0)
    y_preds = (y_preds > 0.5).astype(int)

    # Generate and save submissions
    df_sub = pd.DataFrame()
    df_sub['TARGET_FLAG'] = y_preds

    dir = os.path.split(args.dst)[0]
    if not os.path.exists(dir):
        os.mkdir(dir)
    df_sub.to_csv(args.dst, index=None)