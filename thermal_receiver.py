from fastapi import FastAPI, Request
from datetime import datetime
import json
import os
import threading
from urllib import request as urlrequest
from urllib.error import URLError, HTTPError

import numpy as np
import pandas as pd

app = FastAPI(title="Thermal Receiver API")

SAVE_DIR = "received_data"
DATA_DIR = "data"
RAW_HISTORY_CSV = os.path.join(DATA_DIR, "live_thermal_history.csv")
PREDICTIONS_CSV = os.path.join(DATA_DIR, "live_predictions.csv")
EVALUATIONS_CSV = os.path.join(DATA_DIR, "live_evaluations.csv")
DEFAULT_PREDICTION_CALLBACK_URL = "http://100.68.240.98:7000/api/prediction"
PREDICTION_CALLBACK_URL = os.getenv(
    "PREDICTION_CALLBACK_URL",
    DEFAULT_PREDICTION_CALLBACK_URL,
).strip()
CALLBACK_TIMEOUT_SECONDS = float(os.getenv("PREDICTION_CALLBACK_TIMEOUT", "5"))

with open("model/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

TARGETS = config["targets"]
TARGET_COLS = config["target_cols"]
HORIZON = config["horizon"]
TARGET_TO_PRED_COL = dict(zip(TARGETS, TARGET_COLS))

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

state_lock = threading.Lock()


def to_timestamp_text(value):
    ts = pd.to_datetime(value)
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def thermal_json_to_row(data: dict) -> dict:
    timestamp = data.get("timestamp") or data.get("Timestamp")
    if not timestamp:
        raise ValueError("Thiếu timestamp trong dữ liệu gửi lên.")

    row = {"Timestamp": to_timestamp_text(timestamp)}
    for point in data.get("points", []):
        point_id = str(point.get("id", "")).replace(":", "_")
        if point_id in TARGETS:
            row[point_id] = point.get("temperature")

    missing = [target for target in TARGETS if target not in row]
    if missing:
        raise ValueError(f"Thiếu điểm nhiệt: {missing}")

    return row


def read_csv_if_exists(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def save_history_row(row: dict) -> pd.DataFrame:
    history_df = read_csv_if_exists(RAW_HISTORY_CSV)
    row_df = pd.DataFrame([row])
    history_df = pd.concat([history_df, row_df], ignore_index=True)
    history_df["Timestamp"] = pd.to_datetime(history_df["Timestamp"], errors="coerce")
    history_df = history_df.dropna(subset=["Timestamp"])
    history_df = history_df.drop_duplicates(subset=["Timestamp"], keep="last")
    history_df = history_df.sort_values("Timestamp").reset_index(drop=True)
    history_df["Timestamp"] = history_df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    history_df.to_csv(RAW_HISTORY_CSV, index=False)
    return history_df


def append_rows(path: str, rows: list[dict]) -> None:
    if not rows:
        return

    new_df = pd.DataFrame(rows)
    old_df = read_csv_if_exists(path)
    out_df = pd.concat([old_df, new_df], ignore_index=True)
    out_df.to_csv(path, index=False)


def evaluate_due_predictions(actual_row: dict) -> dict | None:
    predictions_df = read_csv_if_exists(PREDICTIONS_CSV)
    if predictions_df.empty:
        return None

    actual_ts = pd.to_datetime(actual_row["Timestamp"])
    predictions_df["forecast_timestamp_dt"] = pd.to_datetime(
        predictions_df["forecast_timestamp"], errors="coerce"
    )
    due_df = predictions_df[predictions_df["forecast_timestamp_dt"] == actual_ts].copy()
    if due_df.empty:
        return None

    existing_eval_df = read_csv_if_exists(EVALUATIONS_CSV)
    existing_keys = set()
    if not existing_eval_df.empty:
        for _, row in existing_eval_df.iterrows():
            key = (str(row.get("issued_at")), str(row.get("forecast_timestamp")), str(row.get("target")))
            existing_keys.add(key)

    eval_rows = []
    for _, pred_row in due_df.iterrows():
        for target in TARGETS:
            pred_col = f"{target}_pred"
            key = (str(pred_row["issued_at"]), str(pred_row["forecast_timestamp"]), target)
            if key in existing_keys:
                continue

            predicted = float(pred_row[pred_col])
            actual = float(actual_row[target])
            error = actual - predicted
            eval_rows.append({
                "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "issued_at": pred_row["issued_at"],
                "input_timestamp": pred_row["input_timestamp"],
                "forecast_timestamp": pred_row["forecast_timestamp"],
                "target": target,
                "predicted": predicted,
                "actual": actual,
                "error": error,
                "abs_error": abs(error),
                "squared_error": error ** 2,
            })

    append_rows(EVALUATIONS_CSV, eval_rows)
    if not eval_rows:
        return None

    abs_errors = [row["abs_error"] for row in eval_rows]
    squared_errors = [row["squared_error"] for row in eval_rows]
    return {
        "forecast_timestamp": to_timestamp_text(actual_ts),
        "count": len(eval_rows),
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean(squared_errors))),
        "details": eval_rows,
    }


def save_prediction(history_df: pd.DataFrame) -> dict:
    from src.predict import predict_last

    pred_df = predict_last(history_df)
    input_timestamp = pd.to_datetime(pred_df.iloc[0]["Timestamp"])
    forecast_timestamp = input_timestamp + pd.Timedelta(minutes=HORIZON)

    record = {
        "issued_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_timestamp": to_timestamp_text(input_timestamp),
        "forecast_timestamp": to_timestamp_text(forecast_timestamp),
    }

    for target, pred_col in TARGET_TO_PRED_COL.items():
        record[f"{target}_pred"] = float(pred_df.iloc[0][pred_col])

    predictions_df = read_csv_if_exists(PREDICTIONS_CSV)
    if not predictions_df.empty:
        predictions_df = predictions_df[
            predictions_df["input_timestamp"] != record["input_timestamp"]
        ]
    predictions_df = pd.concat([predictions_df, pd.DataFrame([record])], ignore_index=True)
    predictions_df.to_csv(PREDICTIONS_CSV, index=False)

    return record


def send_prediction_callback(prediction: dict) -> dict:
    if not PREDICTION_CALLBACK_URL:
        return {
            "sent": False,
            "reason": "PREDICTION_CALLBACK_URL chưa được cấu hình",
        }

    payload = {
        "type": "thermal_prediction",
        "prediction": prediction,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urlrequest.Request(
        PREDICTION_CALLBACK_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlrequest.urlopen(req, timeout=CALLBACK_TIMEOUT_SECONDS) as resp:
            response_body = resp.read().decode("utf-8", errors="replace")
            return {
                "sent": True,
                "status_code": resp.status,
                "response": response_body,
            }
    except HTTPError as e:
        response_body = e.read().decode("utf-8", errors="replace")
        return {
            "sent": False,
            "status_code": e.code,
            "error": response_body,
        }
    except URLError as e:
        return {
            "sent": False,
            "error": str(e.reason),
        }
    except Exception as e:
        return {
            "sent": False,
            "error": str(e),
        }


def get_latest_prediction() -> dict | None:
    predictions_df = read_csv_if_exists(PREDICTIONS_CSV)
    if predictions_df.empty:
        return None

    predictions_df["issued_at_dt"] = pd.to_datetime(
        predictions_df["issued_at"], errors="coerce"
    )
    predictions_df = predictions_df.sort_values("issued_at_dt").reset_index(drop=True)
    latest = predictions_df.iloc[-1].drop(labels=["issued_at_dt"]).to_dict()

    for key, value in latest.items():
        if pd.isna(value):
            latest[key] = None
        elif key.endswith("_pred"):
            latest[key] = float(value)
        else:
            latest[key] = str(value)

    return latest


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/latest-prediction")
def latest_prediction():
    prediction = get_latest_prediction()
    if not prediction:
        return {
            "success": False,
            "message": "Chưa có prediction nào.",
            "prediction": None,
        }

    return {
        "success": True,
        "prediction": prediction,
    }


@app.post("/api/send-latest-prediction")
def send_latest_prediction():
    prediction = get_latest_prediction()
    if not prediction:
        return {
            "success": False,
            "message": "Chưa có prediction nào để gửi.",
        }

    callback_result = send_prediction_callback(prediction)
    return {
        "success": callback_result.get("sent", False),
        "callback_url": PREDICTION_CALLBACK_URL,
        "prediction": prediction,
        "prediction_callback": callback_result,
    }


@app.post("/api/thermal-data")
async def receive_thermal_data(request: Request):
    try:
        data = await request.json()

        # lưu raw json ra file để kiểm tra
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(SAVE_DIR, f"thermal_{now_str}.json")

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print("=== NHẬN DỮ LIỆU THÀNH CÔNG ===")
        print(json.dumps(data, ensure_ascii=False, indent=2))

        row = thermal_json_to_row(data)
        with state_lock:
            history_df = save_history_row(row)
            evaluation = evaluate_due_predictions(row)

            try:
                prediction = save_prediction(history_df)
                prediction_status = "created"
            except ValueError as e:
                prediction = None
                prediction_status = f"not_ready: {e}"
            except Exception as e:
                prediction = None
                prediction_status = f"failed: {e}"

        callback_result = None

        print(f"Đã lưu history: {len(history_df)} dòng")
        if evaluation:
            print(
                "Đã đánh giá prediction cũ: "
                f"forecast_timestamp={evaluation['forecast_timestamp']}, "
                f"MAE={evaluation['mae']:.4f}, RMSE={evaluation['rmse']:.4f}"
            )
        else:
            print("Chưa có prediction cũ đến hạn để so sánh")

        print(f"Trạng thái dự đoán mới: {prediction_status}")
        if prediction:
            print(json.dumps(prediction, ensure_ascii=False, indent=2))
            print("Prediction đã lưu, chờ bên kia fetch qua GET /api/latest-prediction")

        return {
            "success": True,
            "message": "Đã nhận dữ liệu nhiệt",
            "received_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "actual_row": row,
            "history_rows": len(history_df),
            "evaluation": evaluation,
            "prediction_status": prediction_status,
            "prediction": prediction,
            "prediction_callback": callback_result,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
