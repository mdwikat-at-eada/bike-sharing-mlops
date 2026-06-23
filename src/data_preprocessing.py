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
