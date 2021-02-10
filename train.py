import numpy as np
import pandas as pd
import argparse
import os

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from joblib import dump

def return_preprocessor():
    """
        Return preprocessor pipe object.
    """

    numeric_features = ['AGE',
                        'YOJ',
                        'INCOME',
                        'HOME_VAL',
                        'BLUEBOOK',
                        'OLDCLAIM',
                        'CAR_AGE']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features), ],
        remainder='passthrough')

    return preprocessor

def return_model(args, best_params={}):
    """
    Retourne le modèle specifié lors de l'appel du script.
    """

    if args.model == 'XGB':
        model = XGBClassifier(objective='binary:logistic',
                              use_label_encoder=False,
                              eval_metric='logloss',
                              random_state=args.random_state,
                              **best_params)
    elif args.model == 'RF':
        model = RandomForestClassifier(random_state=args.random_state,
                                       **best_params)
    else:
        raise NotImplementedError("To be implemented")
    return model

def return_param_grid(args):
    """
    Retourne la grille de paramètres correspondant au modèle specifié lors de l'appel du script.
    """

    if args.model == 'XGB':
        param_grid  = dict(clf__n_estimators  = np.linspace(100, 300, 6).astype(int),
                           clf__max_depth     = [5, 8, 10, 15, 20, 30, 50, 75, 100],
                           clf__alpha         = [0.01, 0.05, 0.1, 0.3, 0.5, 1, 10],
                           clf__learning_rate = [0.1, 0.08, 0.05, 0.02, 0.01],
                 )
    elif args.model == 'RF':
        param_grid = dict(clf__n_estimators     = np.linspace(100, 600, 6).astype(int),
                          clf__max_depth        = [None] + list(np.arange(5, 25).astype(int)),
                          clf__max_features     = np.arange(10, 15),
                          clf__min_samples_leaf = np.arange(1, 10),
                          clf__min_samples_split= np.arange(2, 10),
                          )
    else:
        raise NotImplementedError("To be implemented")
    return param_grid

def return_search(args, pipe, param_grid, cv, scoring, n_iter):
    if args.model in ['XGB', 'RF']:
        search = RandomizedSearchCV(pipe,
                                    param_distributions=param_grid,
                                    cv=cv,
                                    verbose=1,
                                    scoring=scoring,
                                    random_state=args.random_state,
                                    n_iter=n_iter,
                                    refit=False
                                    )
    else:
        raise NotImplementedError("To be implemented")
    return search

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default='processed_data/train_for_gridsearch.csv', type=str)
    parser.add_argument("--n_splits", default=5, type=int)
    parser.add_argument("--random_state", default=4, type=int)
    parser.add_argument('--model', default='XGB', choices=['XGB', 'RF'])
    parser.add_argument('--models_dir', default='models', type=str)
    parser.add_argument('--n_iter', default=30, type=int)
    args = parser.parse_args()

    X_for_gridsearch = pd.read_csv(args.src)
    y_for_gridsearch = X_for_gridsearch['TARGET_FLAG'].values
    X_for_gridsearch = X_for_gridsearch.drop(columns=['TARGET_FLAG'])

    # We use k-fold on the remaining data to search over hyper-parameters
    kf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)

    # Model Definition
    preprocessor = return_preprocessor()
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('clf', return_model(args))])
    param_grid = return_param_grid(args)

    search = return_search(args, pipe, param_grid, cv=kf, scoring='f1', n_iter=args.n_iter)

    search.fit(X_for_gridsearch, y_for_gridsearch)
    print('Average f1 score on grid search: {}'.format(search.best_score_))
    best_params = {key[5:]: value for key, value in search.best_params_.items()}

    # Refit and save models for each folds
    models = []
    f1s = []
    for idx_train, idx_test in kf.split(X_for_gridsearch, y_for_gridsearch):
        preprocessor = return_preprocessor()
        clf = return_model(args, best_params)
        pipe = Pipeline(steps=[('preprocessor', preprocessor),
                               ('clf', clf)])

        pipe.fit(X_for_gridsearch.iloc[idx_train], y_for_gridsearch[idx_train])

        y_pred = pipe.predict(X_for_gridsearch.iloc[idx_test])

        models.append(pipe)
        f1s.append(f1_score(y_for_gridsearch[idx_test], y_pred))
    print('Average f1 score on best model (refit - sanity check) : {} +- {}'.format(np.mean(f1s), np.std(f1s)))

    if not os.path.exists(args.models_dir):
        os.mkdir(args.models_dir)

    print('Saving {} models on {}/'.format(args.model, args.models_dir))
    for fold, model in enumerate(models):
        dump(model, os.path.join(args.models_dir, '{}_fold{}.joblib'.format(args.model, fold)))