import os
import math
import re
from decimal import Decimal
import argparse

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from joblib import dump

def replace_dollar_comma_sign(money):
    """
        Given a string representing an amount of money, return the corresponding float representation.
    """
    if type(money) is not str and math.isnan(money):
        return np.nan
    else:
        return float(Decimal(re.sub(r'[^\d.]', '', money)))

def prepare_dollar_columns(df):
    """
        Prepare columns representing amount of money stored in string to float values.
    """
    DOLLARS_COLUMNS = ['INCOME',
                       'HOME_VAL',
                       'BLUEBOOK',
                       'OLDCLAIM'
                       ]

    for column in DOLLARS_COLUMNS:
        df.loc[:, column] = df.loc[:, column].apply(replace_dollar_comma_sign)
        df.loc[:, column] = df.loc[:, column].apply(np.log)
        df.loc[:, column] = df.loc[:, column].replace({-np.inf: 1})
    return df

def prepare_yes_no_columns(df):
    """
        Prepare yes/no binary columns
    """
    BINARY_COLUMNS = ['PARENT1',
                      'MSTATUS',
                      'RED_CAR',
                      'REVOKED']

    replace_binary_dict = {'No': 0,
                           'z_No': 0,
                           'no': 0,
                           'Yes': 1,
                           'yes': 1}

    for column in BINARY_COLUMNS:
        df.loc[:, column].replace(replace_binary_dict,inplace=True)
    return df

def prepare_categorical_variables(df_train, df_test):
    """
        Prepare categorical variables into one-hot-representation.
    """

    CATEGORICAL_COLUMNS = ['SEX',
                          'JOB',
                          'CAR_USE',
                          'CAR_TYPE',
                           'URBANICITY']

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(df_train.loc[:, CATEGORICAL_COLUMNS])

    for df in [df_train, df_test]:
        features = enc.transform(df.loc[:, CATEGORICAL_COLUMNS]).toarray()
        df.drop(columns=CATEGORICAL_COLUMNS, inplace=True)
        new_columns = []
        i = 0
        for category in enc.categories_:
            # We keep only one column for binary categorical features
            if len(category) == 2:
                new_column = category[0]
                df[new_column] = features[:, i]
                new_columns.append(new_column)
                i += 2
            # O.W, we use one-hot-encoding
            else:
                for new_column in category:
                    df[new_column] = features[:, i]
                    new_columns.append(new_column)
                    i += 1

    return df_train, df_test, enc

def input_missing_value(df, label='JOB', by='z_Blue Collar'):
    df.loc[:, label].fillna(by, inplace=True)
    return df

def drop_irrelevant_columns(df):
    """
        Drop irrelevant features.
    """
    DROP_COLUMNS = ['INDEX',
                    'TARGET_AMT']
    df.drop(columns=DROP_COLUMNS, inplace=True)
    return df

def ordinal_education_encoding(df):
    """
        The 'education' column is a bit different since we can define an ordinal ordering between the categories.
        For instance, 'PhD' is a higher level of study than 'Master', which is higher than 'Bachelors', and so on.
        For this reason, it makes sense to map High School to 0, Bachelors to 1, and so on ...
    """
    df.loc[:, 'EDUCATION'].replace({'z_High School': '<High School'}, inplace=True)
    df.loc[:, 'EDUCATION'].replace({'<High School': 0,
                                    'Bachelors': 1,
                                    'Masters': 2,
                                    'PhD': 3}, inplace=True)
    return df

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_src", default='auto-insurance-fall-2017/train_auto.csv', type=str)
    parser.add_argument("--test_dst", default='auto-insurance-fall-2017/test_auto.csv', type=str)
    parser.add_argument("--dst", default='processed_data', type=str)
    parser.add_argument("--models_dir", default='models', type=str)

    args = parser.parse_args()

    # Open dataset
    df_train = pd.read_csv(args.train_src)
    df_test  = pd.read_csv(args.test_dst)

    df_train = prepare_dollar_columns(df_train)
    df_test  = prepare_dollar_columns(df_test)

    df_train = prepare_yes_no_columns(df_train)
    df_test  = prepare_yes_no_columns(df_test)

    df_train = ordinal_education_encoding(df_train)
    df_test  =ordinal_education_encoding(df_test)

    df_train = input_missing_value(df_train)
    df_test  = input_missing_value(df_test)

    df_train, df_test, enc = prepare_categorical_variables(df_train, df_test)

    df_train = drop_irrelevant_columns(df_train)
    df_test  = drop_irrelevant_columns(df_test)

    # Prepare new dirs
    if not os.path.exists(args.dst):
        os.mkdir(args.dst)
    if not os.path.exists(args.models_dir):
        os.mkdir(args.models_dir)

    # Saving CSVs and one hot encoder object
    df_train.to_csv(os.path.join(args.dst, 'train.csv'), index=None)
    df_test.to_csv(os.path.join(args.dst, 'test.csv'), index=None)
    dump(enc, os.path.join(args.models_dir, 'enc'))