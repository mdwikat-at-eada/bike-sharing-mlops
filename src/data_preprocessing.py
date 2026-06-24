# In this file I prepare the data for the model. I load the raw daily data,
# drop the columns I can't use, and build the transformer that turns the 11
# features into numbers the model understands: one-hot encoding for the
# categories and polynomial terms for the weather numbers. I don't train
# anything here - I use these functions later in train.py.

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

from src import config

# categories (their numbers are labels, not amounts) -> one-hot encode
CATEGORICAL = ["season", "mnth", "weekday", "weathersit"]

# continuous numbers -> add squares and interactions (polynomial degree 2)
NUMERIC = ["temp", "atemp", "hum", "windspeed"]


# read the raw csv and drop the columns we don't use
def load_data():
    df = pd.read_csv(config.RAW_DATA_PATH)
    df = df.drop(columns=config.DROP_COLUMNS)
    return df


# split the dataframe into features (X) and target (y = cnt)
def get_X_y(df):
    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET_COLUMN]
    return X, y


# save the clean dataset (the 11 features + cnt, after dropping the unused columns)
# so there is a visible record of the data we actually model on
def save_processed_data(df):
    df.to_csv(config.PROCESSED_DATA_PATH, index=False)


# one-hot encode the categories, add polynomial terms to the numbers,
# and let the rest (yr, holiday, workingday) pass through untouched
def build_preprocessor():
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("num", PolynomialFeatures(degree=2, include_bias=False), NUMERIC),
        ],
        remainder="passthrough",
    )
    return preprocessor
