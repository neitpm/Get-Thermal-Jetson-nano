import json
import joblib
import pandas as pd

from src.model_builder import build_mtkan_model
from src.preprocess import create_input_last_window, create_input_all_windows

with open("model/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

TARGET_COLS = config["target_cols"]
scaler_y = joblib.load("model/scaler_y.pkl")

model = build_mtkan_model("model/model_config_train.json")
model.load_weights("model/mtkan_weights.weights.h5")


def predict_last(data_df: pd.DataFrame) -> pd.DataFrame:
    X, df_prepared = create_input_last_window(data_df)
    y_pred_scaled = model.predict(X, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    result_df = pd.DataFrame(y_pred, columns=TARGET_COLS)
    if "Timestamp" in df_prepared.columns:
        result_df.insert(0, "Timestamp", [df_prepared.iloc[-1]["Timestamp"]])
    return result_df


def predict_all_windows(data_df: pd.DataFrame) -> pd.DataFrame:
    X_all, end_indices, df_prepared = create_input_all_windows(data_df)

    y_pred_scaled = model.predict(X_all, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    result_df = pd.DataFrame(y_pred, columns=TARGET_COLS)
    result_df.insert(0, "window_end_index", end_indices)

    if "Timestamp" in df_prepared.columns:
        result_df.insert(1, "Timestamp", df_prepared.iloc[end_indices]["Timestamp"].values)

    return result_df
