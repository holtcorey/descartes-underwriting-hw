import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default='processed_data/train.csv', type=str)
    parser.add_argument("--dst_for_gridsearch", default='processed_data/train_for_gridsearch.csv', type=str)
    parser.add_argument("--dst_future_val", default='processed_data/train_future_validation.csv', type=str)
    parser.add_argument("--random_state", default=4, type=int)
    args = parser.parse_args()

    df = pd.read_csv(args.src)
    y = df['TARGET_FLAG'].values

    # On garde un jeu de données pour effectuer la recherche d'hyperparamètres
    # On se reserve une partition du jeu de données pour mesurer la généralisation du modèle
    X_for_gridsearch, X_future_validation, y_for_gridsearch, y_future_validation \
        = train_test_split(df, y, test_size=0.2, random_state=args.random_state, stratify=y)

    X_for_gridsearch.to_csv(args.dst_for_gridsearch, index=None)
    X_future_validation.to_csv(args.dst_future_val, index=None)