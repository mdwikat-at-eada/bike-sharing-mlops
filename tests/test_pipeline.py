# Basic test suite for the bike-sharing pipeline.
# These run in CI on every pull request. They check that the data loads and is
# cleaned correctly, that the model pipeline can fit and predict, and that the
# on-demand input file has the columns the model expects. They are fast (a few
# seconds) and need no saved model file.



import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src import config
from src.data_preprocessing import load_data, get_X_y, build_preprocessor
from src.train import make_pipeline


def test_load_data_drops_unused_columns():
    df = load_data()
    # the leakage / id / date columns must be gone
    for col in config.DROP_COLUMNS:
        assert col not in df.columns
    # the 11 features and the target must remain
    for col in config.FEATURE_COLUMNS + [config.TARGET_COLUMN]:
        assert col in df.columns
    assert len(df) == 731


def test_get_X_y_shapes():
    df = load_data()
    X, y = get_X_y(df)
    assert list(X.columns) == config.FEATURE_COLUMNS
    assert X.shape[1] == 11
    assert len(X) == len(y) == len(df)
    assert y.name == config.TARGET_COLUMN


def test_pipeline_fits_and_predicts():
    df = load_data()
    X, y = get_X_y(df)
    pipe = make_pipeline(LinearRegression())
    pipe.fit(X, y)
    preds = pipe.predict(X)
    # one prediction per row, all finite numbers
    assert len(preds) == len(y)
    assert np.isfinite(preds).all()


def test_preprocessor_outputs_numeric_matrix():
    df = load_data()
    X, _ = get_X_y(df)
    pre = build_preprocessor()
    transformed = pre.fit_transform(X)
    # the preprocessor must expand the 11 raw columns into more numeric features
    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > X.shape[1]


def test_on_demand_dataset_has_required_features():
    sample = pd.read_csv(config.BATCH_INPUT_PATH)
    for col in config.FEATURE_COLUMNS:
        assert col in sample.columns
