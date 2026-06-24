# Bike Sharing — Daily Demand Prediction (MLOps)

Predict the **daily number of bike rentals** (`cnt`) from weather and calendar
information, using a simple, interpretable linear regression. The project follows
an MLOps workflow: data preparation, experiment tracking with MLflow, model
selection, and CI/CD that retrains and serves predictions automatically.

> Dataset: [Bike Sharing — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) (`day.csv`, 731 daily records).

---

## The problem

- **Type:** regression (predict a number).
- **Target:** `cnt` — total bikes rented that day.
- **Features used (11):** `season`, `yr`, `mnth`, `holiday`, `weekday`,
  `workingday`, `weathersit`, `temp`, `atemp`, `hum`, `windspeed`.
- **Dropped:** `instant` (id), `dteday` (date), and `casual` + `registered`
  (they sum exactly to `cnt`, so using them would be data leakage).

---

## Project structure

```
bike-sharing-mlops/
├── datasets/
│   └── raw/day.csv                  # raw daily data
├── notebooks/
│   └── 01_eda.ipynb                 # exploratory data analysis
├── src/
│   ├── config.py                    # shared paths, target, feature lists
│   ├── data_preprocessing.py        # load data + feature engineering
│   └── train.py                     # model comparison + MLflow + save best model
├── models/
│   └── best_model.pkl               # trained pipeline (created by training)
├── batch_prediction_dataset/
│   ├── on_demand_dataset.csv        # sample input for batch prediction
│   └── predictions.csv              # prediction output (created by predict.py)
├── report/
│   ├── model_development.md         # model development write-up
│   └── images/                      # charts and MLflow screenshots
├── .github/workflows/               # CI / CD / on-demand prediction
├── main.py                          # entry point: runs the training
├── predict.py                       # batch prediction script
└── requirements.txt
```

---

## How it works

1. **EDA** (`notebooks/01_eda.ipynb`) — explores the data, checks quality, and
   looks at how each feature relates to `cnt`.
2. **Feature engineering** (`src/data_preprocessing.py`) — one-hot encodes the
   categorical columns (`season`, `mnth`, `weekday`, `weathersit`) and adds
   polynomial terms to the continuous weather columns. Everything lives inside a
   single scikit-learn `Pipeline`, so the saved model accepts the raw 11 columns.
3. **Training** (`src/train.py`) — splits the data **70/15/15**
   (train/validation/test), compares **LinearRegression, Ridge and Lasso**
   (logging each run to MLflow), selects the best by validation RMSE, retrains it
   on train+validation, evaluates it once on test, and saves it to
   `models/best_model.pkl`.
4. **Prediction** (`predict.py`) — loads the saved model and predicts on
   `batch_prediction_dataset/on_demand_dataset.csv`, writing `predictions.csv`.
5. **CI/CD** (`.github/workflows/`) — CI runs on pull requests; CD retrains the
   model on push to `main`; the on-demand workflow runs the batch prediction.

---

## How to run it locally

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. train the model (creates models/best_model.pkl and MLflow runs)
python main.py

# 3. run a batch prediction (creates batch_prediction_dataset/predictions.csv)
python predict.py

# 4. (optional) open the MLflow UI to see the experiments
mlflow ui
```

---

## Results

| Model | Validation RMSE | Validation R² |
|---|---|---|
| LinearRegression | 3898 | -2.655 |
| Ridge | 699 | 0.883 |
| **Lasso (selected)** | **698** | **0.883** |

- **Selected model: Lasso** — **test RMSE ≈ 667 bikes/day**, **R² ≈ 0.88**.
- **Key finding:** with the polynomial features the plain LinearRegression
  overfits and gives a negative R²; regularization (Ridge/Lasso) keeps it stable.
  This is the reason we use a regularized linear model.

See [`report/model_development.md`](report/model_development.md) for the full
write-up and the MLflow screenshots.
