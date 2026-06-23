RAW_DATA_PATH = "datasets/raw/day.csv"
PROCESSED_DATA_PATH = "datasets/processed/processed_day.csv"
MODEL_PATH = "models/best_model.pkl"
BATCH_INPUT_PATH = "batch_prediction_dataset/on_demand_dataset.csv"
BATCH_OUTPUT_PATH = "batch_prediction_dataset/predictions.csv"
TARGET_COLUMN = "cnt"

DROP_COLUMNS = [
"instant",
"dteday",
"casual",
"registered"
]

FEATURE_COLUMNS = [
"season",
"yr",
"mnth",
"holiday",
"weekday",
"workingday",
"weathersit",
"temp",
"atemp",
"hum",
"windspeed"
]