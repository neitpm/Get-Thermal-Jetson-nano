from fastapi import FastAPI, Request
import uvicorn
from datetime import datetime
import json
import os
import threading
import subprocess
import sys
import time
import requests
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
DEFAULT_PREDICTION_CALLBACK_URL = "http://192.168.10.102:8100/api/prediction"
PREDICTION_CALLBACK_URL = os.getenv(
    "PREDICTION_CALLBACK_URL",
    DEFAULT_PREDICTION_CALLBACK_URL,
).strip()
CALLBACK_TIMEOUT_SECONDS = float(os.getenv("PREDICTION_CALLBACK_TIMEOUT", "5"))

TARGETS = []
TARGET_COLS = []
HORIZON = 5
TARGET_TO_PRED_COL = {}

def reload_config():
    global TARGETS, TARGET_COLS, HORIZON, TARGET_TO_PRED_COL, _cached_history_df, _cached_predictions_df
    try:
        with open("model/config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        TARGETS = config["targets"]
        TARGET_COLS = config["target_cols"]
        HORIZON = config["horizon"]
        TARGET_TO_PRED_COL = dict(zip(TARGETS, TARGET_COLS))
        _cached_history_df = None
        _cached_predictions_df = None
        print(f"[CONFIG] Da reload cấu hình. Targets mới: {TARGETS}")
    except Exception as e:
        print(f"[CONFIG-ERROR] Lỗi khi reload cấu hình: {e}")

# Load ban đầu
reload_config()


os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

state_lock = threading.Lock()


# Global variables to cache data for performance
_cached_history_df = None
_cached_predictions_df = None


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

    # Điền giá trị None cho các target bị thiếu (ví dụ: chưa có tọa độ hoặc lỗi cảm biến)
    for target in TARGETS:
        if target not in row:
            row[target] = None

    return row


def read_csv_if_exists(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def save_history_row(row: dict) -> pd.DataFrame:
    global _cached_history_df
    
    # 1. Fast append to CSV
    row_df = pd.DataFrame([row])
    file_exists = os.path.exists(RAW_HISTORY_CSV)
    row_df.to_csv(RAW_HISTORY_CSV, mode='a', header=not file_exists, index=False)
    
    # 2. Update memory cache (keep only last 200 rows for prediction)
    if _cached_history_df is None:
        if file_exists:
            _cached_history_df = pd.read_csv(RAW_HISTORY_CSV).tail(200)
        else:
            _cached_history_df = row_df.copy()
    else:
        _cached_history_df = pd.concat([_cached_history_df, row_df], ignore_index=True)
        if len(_cached_history_df) > 200:
            _cached_history_df = _cached_history_df.tail(200)
            
    return _cached_history_df


def append_rows(path: str, rows: list[dict]) -> None:
    if not rows:
        return

    new_df = pd.DataFrame(rows)
    old_df = read_csv_if_exists(path)
    out_df = pd.concat([old_df, new_df], ignore_index=True)
    out_df.to_csv(path, index=False)


def evaluate_due_predictions(actual_row: dict) -> dict | None:
    global _cached_predictions_df
    
    if _cached_predictions_df is None:
        _cached_predictions_df = read_csv_if_exists(PREDICTIONS_CSV).tail(100)
        
    if _cached_predictions_df.empty:
        return None

    actual_ts = pd.to_datetime(actual_row["Timestamp"])
    df = _cached_predictions_df.copy()
    df["forecast_timestamp_dt"] = pd.to_datetime(
        df["forecast_timestamp"], errors="coerce"
    )
    due_df = df[df["forecast_timestamp_dt"] == actual_ts].copy()
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
    global _cached_predictions_df
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

    # 1. Fast append to CSV
    row_df = pd.DataFrame([record])
    file_exists = os.path.exists(PREDICTIONS_CSV)
    row_df.to_csv(PREDICTIONS_CSV, mode='a', header=not file_exists, index=False)
    
    # 2. Update cache
    if _cached_predictions_df is None:
        if file_exists:
            _cached_predictions_df = pd.read_csv(PREDICTIONS_CSV).tail(100)
        else:
            _cached_predictions_df = row_df.copy()
    else:
        _cached_predictions_df = pd.concat([_cached_predictions_df, row_df], ignore_index=True)
        if len(_cached_predictions_df) > 100:
            _cached_predictions_df = _cached_predictions_df.tail(100)

    return record


def send_prediction_callback(prediction: dict, target_url: str = None) -> dict:
    url = target_url or PREDICTION_CALLBACK_URL
    if not url:
        return {
            "sent": False,
            "reason": "URL callback chưa được cấu hình",
        }

    payload = prediction
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urlrequest.Request(
        url,
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
@app.get("/api/prediction")
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


# ==========================================
# AUTOMATIC BACKGROUND RETRAINING DAEMON
# ==========================================
training_lock = threading.Lock()
is_training = False

def trigger_background_training():
    global is_training
    with training_lock:
        if is_training:
            print("[AUTOTRAIN] Một tiến trình huấn luyện đang chạy, bỏ qua kích hoạt trùng lặp.", flush=True)
            return
        is_training = True

    def train_job():
        global is_training
        print("[AUTOTRAIN] Bắt đầu tự động huấn luyện lại model (Fine-tuning)...", flush=True)
        try:
            cwd = os.path.dirname(os.path.abspath(__file__))
            venv_python = os.path.join(cwd, "tf_env", "bin", "python")
            if not os.path.exists(venv_python):
                venv_python = sys.executable

            p = subprocess.Popen(
                [venv_python, "train.py"],
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            # Đọc liên tục log để tránh đầy buffer ống dẫn
            for line in iter(p.stdout.readline, ''):
                if line:
                    print(f"[AUTOTRAIN-LOG] {line.strip()}", flush=True)
            p.wait()
            print(f"[AUTOTRAIN] Đã hoàn thành tự động huấn luyện lại model. Exit code: {p.returncode}", flush=True)
            
            if p.returncode == 0:
                # Nạp lại trọng số mới vào RAM trực tiếp không cần khởi động lại
                from src.predict import reload_model_weights
                reload_model_weights()
                
                # Ghi nhận mốc thời gian huấn luyện thành công
                meta_path = os.path.join(cwd, "model", "training_meta.json")
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump({"last_trained": time.time()}, f, ensure_ascii=False, indent=4)
                    
        except Exception as e:
            print(f"[AUTOTRAIN-ERROR] Lỗi khi chạy huấn luyện ngầm: {e}", flush=True)
        finally:
            with training_lock:
                is_training = False

    threading.Thread(target=train_job, daemon=True).start()


def check_and_trigger_autotrain():
    """Tự động kiểm tra xem đã qua 2 ngày (172800 giây) kể từ lần train cuối cùng chưa."""
    cwd = os.path.dirname(os.path.abspath(__file__))
    meta_path = os.path.join(cwd, "model", "training_meta.json")
    should_train = False
    
    # 2 ngày = 2 * 24 * 3600 = 172800 giây
    INTERVAL_SECONDS = 172800
    
    if not os.path.exists(meta_path):
        # Chưa từng train tự động -> Đánh dấu mốc ban đầu (không train ngay lập tức để tránh overload lúc mới chạy)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"last_trained": time.time()}, f, ensure_ascii=False, indent=4)
        return

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        last_trained = meta.get("last_trained", 0)
        if time.time() - last_trained >= INTERVAL_SECONDS:
            should_train = True
    except Exception:
        should_train = True

    if should_train:
        trigger_background_training()


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
        prediction = None
        prediction_status = "not_started"
        callback_result = None

        callback_result = None
        with state_lock:
            history_df = save_history_row(row)
            evaluation = evaluate_due_predictions(row)

            print(f"Dữ liệu hiện có: {len(history_df)} dòng trong lịch sử.")
            try:
                prediction = save_prediction(history_df)
                prediction_status = "created"
                print("Dự đoán thành công!")
            except ValueError as e:
                prediction_status = f"not_ready: {e}"
                print(f"Chưa thể dự đoán: {e}")
            except Exception as e:
                prediction_status = f"failed: {e}"
                print(f"Lỗi khi dự đoán: {e}")
        # Thực hiện cơ chế PUSH dự báo sang Server trung tâm (hoặc bên thứ 3)
        if prediction:
            target_url = PREDICTION_CALLBACK_URL
            if not target_url:
                client_ip = request.client.host
                target_url = f"http://{client_ip}:8080/api/prediction"
            
            print(f"Đang gửi dự đoán tới {target_url}...")
            callback_result = send_prediction_callback(prediction, target_url=target_url)
            
            if callback_result.get("sent"):
                print(f"Gửi thành công tới {target_url}")
            else:
                print(f"Gửi thất bại tới {target_url}: {callback_result.get('error') or callback_result.get('reason')}")

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
        # Tự động kiểm tra và kích hoạt huấn luyện lại model sau mỗi 2 ngày
        try:
            check_and_trigger_autotrain()
        except Exception as ex:
            print(f"[AUTOTRAIN-ERROR] Lỗi khi kiểm tra tự động huấn luyện: {ex}", flush=True)

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

@app.post("/api/v1/config/thermal")
@app.post("/config/thermal")
async def save_thermal_config(request: Request, forward: bool = True):
    """
    1. API cập nhật tọa độ hình ảnh (Thermal Analyzer Configuration):
    """
    try:
        data = await request.json()
        config_path = os.path.join(DATA_DIR, "thermal_points.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"[CONFIG] Đã nhận và lưu cấu hình điểm đo nhiệt mới vào: {config_path}")

        # 1a. Forward cấu hình sang Server trung tâm (192.168.10.102:8100)
        if forward:
            try:
                central_url = "http://192.168.10.102:8100/config/thermal"
                print(f"[CONFIG] Forwarding config/thermal to central server: {central_url}")
                requests.post(central_url, json=data, timeout=5)
            except Exception as e:
                print(f"[CONFIG-ERROR] Failed to forward config/thermal to central server: {e}")

        # 1b. Tự động đồng bộ targets từ các points vừa nhận vào config.json
        if "points" in data:
            targets = [p.get("label") or p.get("id") for p in data["points"] if p.get("label") or p.get("id")]
            if targets:
                config_path_json = "model/config.json"
                if os.path.exists(config_path_json):
                    with open(config_path_json, "r", encoding="utf-8") as f:
                        model_config = json.load(f)
                else:
                    model_config = {"window_size": 5, "horizon": 5, "seasonal_period": 60}

                model_config["targets"] = targets
                
                # Tạo động feature_cols và target_cols
                horizon = model_config.get("horizon", 5)
                feature_cols = []
                target_cols = []
                for t in targets:
                    feature_cols.extend([t, f"{t}_trend", f"{t}_seasonal", f"{t}_residual"])
                    target_cols.append(f"{t}_t{horizon}")
                    
                model_config["feature_cols"] = feature_cols
                model_config["target_cols"] = target_cols

                with open(config_path_json, "w", encoding="utf-8") as f:
                    json.dump(model_config, f, ensure_ascii=False, indent=4)

                # Làm sạch file raw history (lưu header mới)
                headers = ["Timestamp"] + targets
                pd.DataFrame(columns=headers).to_csv(RAW_HISTORY_CSV, index=False)

                # Làm sạch predictions và evaluations để đồng bộ cấu trúc mới
                pred_headers = ["issued_at", "input_timestamp", "forecast_timestamp"] + [f"{t}_pred" for t in targets]
                pd.DataFrame(columns=pred_headers).to_csv(PREDICTIONS_CSV, index=False)
                
                eval_headers = ["evaluated_at", "issued_at", "input_timestamp", "forecast_timestamp", "target", "predicted", "actual", "error", "abs_error", "squared_error"]
                pd.DataFrame(columns=eval_headers).to_csv(EVALUATIONS_CSV, index=False)

                # Nạp lại cấu hình vào bộ nhớ RAM
                reload_config()

                print(f"[CONFIG] Đã tự động đồng bộ targets từ points mới: {targets}. Kích hoạt Retrain...")
                trigger_background_training()

        return {"success": True, "message": "Đã cập nhật tọa độ điểm nhiệt thành công"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/config/update")
async def update_forecasting_targets(request: Request, forward: bool = True):
    """
    2. API cập nhật danh sách dự báo (Forecasting Targets):
    """
    try:
        data = await request.json()
        targets = data.get("targets")
        if not targets:
            return {"success": False, "error": "Thiếu danh sách targets"}

        # 2a. Forward cấu hình sang Server trung tâm (192.168.10.102:8100)
        if forward:
            try:
                central_url = "http://192.168.10.102:8100/api/config/update"
                print(f"[CONFIG] Forwarding api/config/update to central server: {central_url}")
                requests.post(central_url, json=data, timeout=5)
            except Exception as e:
                print(f"[CONFIG-ERROR] Failed to forward api/config/update to central server: {e}")

        # Đọc config.json hiện tại
        config_path = "model/config.json"
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                model_config = json.load(f)
        else:
            model_config = {"window_size": 5, "horizon": 5, "seasonal_period": 60}

        model_config["targets"] = targets
        
        # Tạo động feature_cols và target_cols
        horizon = model_config.get("horizon", 5)
        feature_cols = []
        target_cols = []
        for t in targets:
            feature_cols.extend([t, f"{t}_trend", f"{t}_seasonal", f"{t}_residual"])
            target_cols.append(f"{t}_t{horizon}")
            
        model_config["feature_cols"] = feature_cols
        model_config["target_cols"] = target_cols

        # Ghi đè targets vào file config
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(model_config, f, ensure_ascii=False, indent=4)

        # Làm sạch file raw history (lưu header mới)
        headers = ["Timestamp"] + targets
        pd.DataFrame(columns=headers).to_csv(RAW_HISTORY_CSV, index=False)

        # Làm sạch predictions và evaluations để đồng bộ cấu trúc mới
        pred_headers = ["issued_at", "input_timestamp", "forecast_timestamp"] + [f"{t}_pred" for t in targets]
        pd.DataFrame(columns=pred_headers).to_csv(PREDICTIONS_CSV, index=False)
        
        eval_headers = ["evaluated_at", "issued_at", "input_timestamp", "forecast_timestamp", "target", "predicted", "actual", "error", "abs_error", "squared_error"]
        pd.DataFrame(columns=eval_headers).to_csv(EVALUATIONS_CSV, index=False)

        # Nạp lại cấu hình vào bộ nhớ RAM
        reload_config()

        print(f"[CONFIG] Đã cập nhật targets dự báo mới: {targets}. Làm sạch dữ liệu cũ và kích hoạt Retrain...")

        # Kích hoạt tiến trình Retrain ngầm
        trigger_background_training()

        return {"success": True, "message": "Cấu hình targets mới thành công và đang retrain ngầm"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/prediction/history")
def get_prediction_history():
    """
    Trả về dữ liệu lịch sử thực tế song song với dữ liệu dự báo cho từng mốc thời gian.
    Format: { "history": [ { "timestamp": "HH:MM", "Nhãn_1_actual": 31.5, "Nhãn_1_pred": 32.0, ... } ] }
    """
    try:
        history_df = read_csv_if_exists(RAW_HISTORY_CSV)
        predictions_df = read_csv_if_exists(PREDICTIONS_CSV)

        timestamps = set()
        if not history_df.empty and "Timestamp" in history_df.columns:
            timestamps.update(history_df["Timestamp"].dropna().tolist())
        if not predictions_df.empty and "forecast_timestamp" in predictions_df.columns:
            timestamps.update(predictions_df["forecast_timestamp"].dropna().tolist())

        sorted_timestamps = sorted(list(timestamps))

        history_lookup = {}
        if not history_df.empty:
            for _, r in history_df.iterrows():
                ts = r.get("Timestamp")
                if ts:
                    history_lookup[str(ts)] = r.to_dict()

        pred_lookup = {}
        if not predictions_df.empty:
            for _, r in predictions_df.iterrows():
                ts = r.get("forecast_timestamp")
                if ts:
                    pred_lookup[str(ts)] = r.to_dict()

        history_list = []
        # Lấy tối đa 50 mốc thời gian gần nhất để vẽ biểu đồ
        for ts in sorted_timestamps[-50:]:
            try:
                dt = pd.to_datetime(ts)
                time_str = dt.strftime("%H:%M")
            except Exception:
                time_str = str(ts)

            entry = {"timestamp": time_str}
            hist_row = history_lookup.get(ts, {})
            pred_row = pred_lookup.get(ts, {})

            for target in TARGETS:
                # Actual
                val_act = hist_row.get(target)
                if pd.isna(val_act) or val_act is None:
                    val_act = None
                else:
                    val_act = float(val_act)
                entry[f"{target}_actual"] = val_act

                # Predicted
                val_pred = pred_row.get(f"{target}_pred")
                if pd.isna(val_pred) or val_pred is None:
                    val_pred = None
                else:
                    val_pred = float(val_pred)
                entry[f"{target}_pred"] = val_pred

            history_list.append(entry)

        return {"history": history_list}
    except Exception as e:
        return {"history": [], "error": str(e)}

@app.get("/api/training-status")
def get_training_status():
    """
    Trả về trạng thái tiến trình huấn luyện model ngầm
    """
    global is_training
    cwd = os.path.dirname(os.path.abspath(__file__))
    meta_path = os.path.join(cwd, "model", "training_meta.json")
    last_trained_str = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            last_trained_time = meta.get("last_trained", 0)
            last_trained_str = datetime.fromtimestamp(last_trained_time).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass

    return {
        "status": "training" if is_training else "idle",
        "last_trained": last_trained_str
    }


if __name__ == "__main__":
    uvicorn.run("thermal_receiver:app", host="0.0.0.0", port=8080, reload=True)
