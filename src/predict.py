import os
import json
import joblib
import pandas as pd
import numpy as np

# Load configuration dynamically
def load_current_config():
    with open("model/config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg

def reload_model_weights():
    # Giữ lại để đảm bảo tính tương thích với hàm gọi từ main/daemon
    return True

def predict_last(data_df: pd.DataFrame) -> pd.DataFrame:
    cfg = load_current_config()
    targets = cfg["targets"]
    target_cols = cfg["target_cols"]
    horizon = cfg["horizon"]
    window_size = cfg.get("window_size", 5)
    
    target_to_pred_col = dict(zip(targets, target_cols))
    predictions = {}
    
    # Thử load mô hình Linear Regression đã được lưu từ file pkl
    lr_models = None
    lr_path = "model/linear_reg.pkl"
    if os.path.exists(lr_path):
        try:
            lr_models = joblib.load(lr_path)
        except Exception as e:
            print(f"[PREDICT] Lỗi load linear regression model: {e}", flush=True)
            
    for target in targets:
        pred_col = target_to_pred_col[target]
        
        # Helper function to get float or default
        def get_safe_last_val(df, col, default=30.0):
            if len(df) == 0:
                return default
            val = df[col].iloc[-1]
            try:
                return float(val) if pd.notna(val) else default
            except:
                return default
        
        # Nếu chưa đủ 2 điểm dữ liệu, giữ nguyên giá trị gần nhất
        if len(data_df) < 2:
            predictions[pred_col] = get_safe_last_val(data_df, target)
            continue
            
        # Thử sử dụng mô hình Autoregressive Linear Regression đã huấn luyện
        if lr_models and target in lr_models:
            try:
                model_lr = lr_models[target]
                # Lấy 5 giá trị gần nhất của target làm đầu vào (lags)
                lags = data_df[target].tail(window_size).values
                # Chuyển đổi lags sang float, thay thế NaN bằng safe value
                lags_clean = []
                for v in lags:
                    try:
                        lags_clean.append(float(v) if pd.notna(v) else 30.0)
                    except:
                        lags_clean.append(30.0)
                lags_clean = np.array(lags_clean)
                
                if len(lags_clean) == window_size:
                    pred_val = float(model_lr.predict([lags_clean])[0])
                    predictions[pred_col] = pred_val
                    continue
            except Exception as e:
                print(f"[PREDICT] Lỗi khi chạy LR model cho {target}: {e}", flush=True)
                
        # Phương án dự phòng (Fallback): Khớp xu hướng tuyến tính (Linear Trend Extrapolation) trên 10 điểm gần nhất
        try:
            # Lọc bỏ giá trị không hợp lệ
            subset = data_df[target].tail(10).dropna()
            subset_clean = []
            for v in subset.values:
                try:
                    val_f = float(v)
                    if pd.notna(val_f):
                        subset_clean.append(val_f)
                except:
                    pass
            
            if len(subset_clean) >= 2:
                y = np.array(subset_clean)
                x = np.arange(len(y))
                slope, intercept = np.polyfit(x, y, 1)
                # Dự báo thời điểm tương lai (tương ứng với +horizon bước)
                pred_val = float(slope * (len(y) - 1 + horizon) + intercept)
                predictions[pred_col] = pred_val
            else:
                predictions[pred_col] = get_safe_last_val(data_df, target)
        except Exception as e:
            print(f"[PREDICT] Lỗi khi tính trend tuyến tính cho {target}: {e}", flush=True)
            predictions[pred_col] = get_safe_last_val(data_df, target)
            
    # Tạo DataFrame chứa kết quả dự đoán
    result_df = pd.DataFrame([predictions])
    if "Timestamp" in data_df.columns and len(data_df) > 0:
        result_df.insert(0, "Timestamp", [data_df.iloc[-1]["Timestamp"]])
    else:
        result_df.insert(0, "Timestamp", [pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")])
        
    return result_df

def predict_all_windows(data_df: pd.DataFrame) -> pd.DataFrame:
    # Hàm này dùng để phân tích offline/đánh giá, ta trả về DataFrame rỗng nếu không dùng
    # nhằm tránh lỗi tương thích khi gọi
    cfg = load_current_config()
    targets = cfg["targets"]
    target_cols = cfg["target_cols"]
    return pd.DataFrame(columns=["window_end_index", "Timestamp"] + target_cols)
