# This is my training script. I split the data 70/15/15, then I compare three
# linear models (LinearRegression, Ridge and Lasso) and log every run to MLflow.
# I pick the best one by validation RMSE, retrain it on train+validation, check
# it on the test set, and save the winning pipeline to models/best_model.pkl
# so the prediction script can use it.

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src import config
from src.data_preprocessing import load_data, get_X_y, build_preprocessor


# the alphas (regularization strength) we try for Ridge and Lasso
ALPHAS = [0.01, 0.1, 1, 10, 100]


# build the pipeline: one-hot encoding + the model, together
def make_pipeline(model):
    return Pipeline([("pre", build_preprocessor()), ("model", model)])


# RMSE of a fitted pipeline on some data
def rmse_of(pipe, X, y):
    return np.sqrt(mean_squared_error(y, pipe.predict(X)))


# quietly try every alpha and return the best one (lowest validation RMSE)
# extra is for extra settings, like max_iter for Lasso
def best_alpha_for(model_class, X_train, y_train, X_val, y_val, **extra):
    best_alpha = None
    best_rmse = None
    for alpha in ALPHAS:
        pipe = make_pipeline(model_class(alpha=alpha, **extra))
        pipe.fit(X_train, y_train)
        rmse = rmse_of(pipe, X_val, y_val)
        if best_rmse is None or rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha
    return best_alpha


# train a model, log it to MLflow as one run, and return its validation RMSE
def log_model(name, model, X_train, y_train, X_val, y_val):
    pipe = make_pipeline(model)
    with mlflow.start_run(run_name=name):
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, pred))
        mae = mean_absolute_error(y_val, pred)
        r2 = r2_score(y_val, pred)

        mlflow.log_param("model", name)
        if hasattr(model, "alpha"):
            mlflow.log_param("alpha", model.alpha)   # only Ridge/Lasso have alpha
        mlflow.log_metric("val_rmse", rmse)
        mlflow.log_metric("val_mae", mae)
        mlflow.log_metric("val_r2", r2)
        mlflow.sklearn.log_model(pipe, "model")

    print(f"{name}: RMSE={rmse:.0f}  MAE={mae:.0f}  R2={r2:.3f}")
    return rmse


def train_model():
    # 1. load data and split 70 / 15 / 15 (train / validation / test)
    df = load_data()
    X, y = get_X_y(df)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    mlflow.set_experiment("bike-sharing")

    # 2. find the best alpha for Ridge and Lasso (quiet search, not logged)
    ridge_alpha = best_alpha_for(Ridge, X_train, y_train, X_val, y_val)
    lasso_alpha = best_alpha_for(Lasso, X_train, y_train, X_val, y_val, max_iter=10000)
    print("Best Ridge alpha:", ridge_alpha, "| Best Lasso alpha:", lasso_alpha)

    # 3. the 3 models we compare (only these get logged to MLflow)
    # FINDING: with the polynomial features, plain LinearRegression overfits and
    # gives a NEGATIVE R2 (it goes unstable with so many correlated features).
    # Ridge and Lasso stay stable and improve -> this is exactly why we regularize.
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=ridge_alpha),
        "Lasso": Lasso(alpha=lasso_alpha, max_iter=10000),
    }
    results = {}
    for name, model in models.items():
        results[name] = log_model(name, model, X_train, y_train, X_val, y_val)

    # 4. the best model = the one with the lowest validation RMSE
    best_name = min(results, key=results.get)
    print("\nBest model:", best_name)

    # 5. retrain the best model with train + val (85%) and check it on test
    final_pipe = make_pipeline(models[best_name])
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    final_pipe.fit(X_train_val, y_train_val)
    print("TEST RMSE:", round(rmse_of(final_pipe, X_test, y_test)))

    # 6. save the final model so predict.py can load it
    joblib.dump(final_pipe, config.MODEL_PATH)
    print("Saved model to", config.MODEL_PATH)


if __name__ == "__main__":
    train_model()
