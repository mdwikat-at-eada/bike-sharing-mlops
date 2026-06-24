import pandas as pd
import joblib
from src.config import BATCH_INPUT_PATH, BATCH_OUTPUT_PATH, MODEL_PATH, FEATURE_COLUMNS

def run_prediction():
    data = pd.read_csv(BATCH_INPUT_PATH)
    model = joblib.load(MODEL_PATH)

    X = data[FEATURE_COLUMNS]
    predictions = model.predict(X)

    data["predicted_cnt"] = predictions
    data.to_csv(BATCH_OUTPUT_PATH, index=False)

    print("Predictions saved successfully.")

if __name__ == "__main__":
    run_prediction()