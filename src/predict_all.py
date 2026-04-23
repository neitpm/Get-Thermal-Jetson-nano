import json
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# =========================
# LOAD CONFIG
# =========================
with open("model/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

WINDOW_SIZE = config["window_size"]
FEATURE_COLS = config["feature_cols"]
TARGET_COLS = config["target_cols"]

# =========================
# LOAD MODEL + SCALER
# =========================
model = load_model("model/mtkan_model.keras", compile=False)
scaler_X = joblib.load("model/scaler_X.pkl")
scaler_y = joblib.load("model/scaler_y.pkl")


def predict_all_windows(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Chạy dự đoán theo sliding window trên toàn bộ file.

    Input:
        data_df: DataFrame chứa đầy đủ các cột feature_cols

    Output:
        DataFrame gồm:
        - window_end_index: vị trí kết thúc của cửa sổ
        - các cột dự đoán tương ứng TARGET_COLS
    """
    # Kiểm tra thiếu cột
    missing_cols = [col for col in FEATURE_COLS if col not in data_df.columns]
    if missing_cols:
        raise ValueError(f"Thiếu các cột đầu vào: {missing_cols}")

    # Kiểm tra số dòng
    if len(data_df) < WINDOW_SIZE:
        raise ValueError(
            f"Số dòng dữ liệu phải >= WINDOW_SIZE ({WINDOW_SIZE}), "
            f"nhưng hiện tại chỉ có {len(data_df)} dòng."
        )

    # Lấy feature
    X_raw = data_df[FEATURE_COLS].values

    # Scale toàn bộ feature
    X_scaled = scaler_X.transform(X_raw)

    all_preds = []
    all_end_idx = []

    # Sliding window
    for end_idx in range(WINDOW_SIZE, len(X_scaled) + 1):
        start_idx = end_idx - WINDOW_SIZE

        X_window = X_scaled[start_idx:end_idx]              # shape: (window_size, n_features)
        X_window = X_window.reshape(1, WINDOW_SIZE, -1)     # shape: (1, window_size, n_features)

        y_pred_scaled = model.predict(X_window, verbose=0)  # shape: (1, n_targets)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)  # đưa về giá trị gốc

        all_preds.append(y_pred[0])
        all_end_idx.append(end_idx - 1)  # index dòng cuối của cửa sổ

    # Tạo DataFrame kết quả
    result_df = pd.DataFrame(all_preds, columns=TARGET_COLS)
    result_df.insert(0, "window_end_index", all_end_idx)

    return result_df


if __name__ == "__main__":
    # Ví dụ test
    input_csv = "data/latest.csv"
    output_csv = "data/predictions_all_windows.csv"

    df = pd.read_csv(input_csv)
    pred_df = predict_all_windows(df)

    print(pred_df.head())
    pred_df.to_csv(output_csv, index=False)
    print(f"Đã lưu kết quả vào: {output_csv}")
