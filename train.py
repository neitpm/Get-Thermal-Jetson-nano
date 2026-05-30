import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# =========================
# CONFIG & PATHS
# =========================
CONFIG_PATH = "model/config.json"
MODEL_CONFIG_PATH = "model/model_config_train.json"
WEIGHTS_PATH = "model/mtkan_weights.weights.h5"
HISTORY_PATH = "data/live_thermal_history.csv"
LR_MODEL_PATH = "model/linear_reg.pkl"

def main():
    print("=" * 70)
    print("           HUẤN LUYỆN LẠI MODEL DỰ BÁO NHIỆT ĐỘ (RETRAIN)            ")
    print("=" * 70)

    # 1. Load Cấu hình
    if not os.path.exists(CONFIG_PATH):
        print(f"[ERROR] Không tìm thấy file cấu hình: {CONFIG_PATH}")
        return

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    WINDOW_SIZE = config.get("window_size", 5)
    HORIZON = config.get("horizon", 5)
    TARGETS = config["targets"]

    print(f"[CONFIG] Window Size: {WINDOW_SIZE} | Horizon: {HORIZON}")
    print(f"[CONFIG] Targets ({len(TARGETS)}): {TARGETS}")

    # 2. Đọc dữ liệu lịch sử nhiệt độ
    if not os.path.exists(HISTORY_PATH):
        print(f"[ERROR] Không tìm thấy file dữ liệu lịch sử: {HISTORY_PATH}")
        # Tạo file trống nếu chưa tồn tại
        headers = ["Timestamp"] + TARGETS
        pd.DataFrame(columns=headers).to_csv(HISTORY_PATH, index=False)
        df_raw = pd.DataFrame(columns=headers)
    else:
        df_raw = pd.read_csv(HISTORY_PATH)
        
    print(f"[DATA] Đọc thành công {len(df_raw)} dòng dữ liệu từ {HISTORY_PATH}")

    # 3. Huấn luyện Linear Regression Autoregressive Model cho từng Target
    lr_models = {}
    print("[TRAIN-LR] Bắt đầu huấn luyện Linear Regression cho các nhãn mới...")
    
    # Số lượng dòng tối thiểu cần thiết để tạo lag features: WINDOW_SIZE + HORIZON
    min_len_required = WINDOW_SIZE + HORIZON
    
    if len(df_raw) >= min_len_required:
        for target in TARGETS:
            try:
                # Tạo X (các lag) và y (nhãn dịch chuyển)
                X_list = []
                y_list = []
                series = df_raw[target].dropna().values
                
                if len(series) >= min_len_required:
                    for i in range(len(series) - WINDOW_SIZE - HORIZON + 1):
                        X_list.append(series[i : i + WINDOW_SIZE])
                        y_list.append(series[i + WINDOW_SIZE + HORIZON - 1])
                    
                    X_arr = np.array(X_list)
                    y_arr = np.array(y_list)
                    
                    # Train model Linear Regression
                    lr_model = LinearRegression()
                    lr_model.fit(X_arr, y_arr)
                    lr_models[target] = lr_model
                    print(f"[TRAIN-LR SUCCESS] Đã lưu LR model cho target: {target}")
            except Exception as e:
                print(f"[TRAIN-LR ERROR] Lỗi khi huấn luyện LR cho {target}: {e}")
                
        # Lưu các model Linear Regression thành file pkl
        if lr_models:
            joblib.dump(lr_models, LR_MODEL_PATH)
            print(f"[SAVE-LR] Đã ghi đè {len(lr_models)} model Linear Regression vào {LR_MODEL_PATH}")
    else:
        print(f"[TRAIN-LR WARNING] Dữ liệu lịch sử hiện tại ({len(df_raw)} dòng) ít hơn mức tối thiểu ({min_len_required} dòng) để huấn luyện Linear Regression. Sử dụng chế độ Extrapolation tuyến tính tự động lúc dự đoán.")

    # 4. Huấn luyện thử nghiệm MTKAN (nếu có đủ dữ liệu lớn)
    if len(df_raw) >= 100:
        try:
            print("[TRAIN-MTKAN] Đang chuẩn bị huấn luyện lại MTKAN...")
            from tensorflow import keras
            from src.preprocess import prepare_feature_df
            from src.model_builder import build_mtkan_model
            
            # Chuẩn hóa & Trích xuất đặc trưng (Features STL)
            df_features = prepare_feature_df(df_raw)
            FEATURE_COLS = config["feature_cols"]
            
            # Load Scalers nếu có, nếu chưa có tạo mới
            try:
                scaler_X = joblib.load("model/scaler_X.pkl")
                scaler_y = joblib.load("model/scaler_y.pkl")
            except Exception:
                from sklearn.preprocessing import StandardScaler
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()
                scaler_X.fit(df_features[FEATURE_COLS].values)
                scaler_y.fit(df_features[TARGETS].values)
                joblib.dump(scaler_X, "model/scaler_X.pkl")
                joblib.dump(scaler_y, "model/scaler_y.pkl")
            
            X_raw = df_features[FEATURE_COLS].values
            y_raw = df_features[TARGETS].values
            
            X_scaled = scaler_X.transform(X_raw)
            y_scaled = scaler_y.transform(y_raw)
            
            X_train = []
            y_train = []
            for i in range(len(X_scaled) - WINDOW_SIZE - HORIZON + 1):
                window = X_scaled[i : i + WINDOW_SIZE]
                target = y_scaled[i + WINDOW_SIZE + HORIZON - 1]
                X_train.append(window)
                y_train.append(target)
                
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            model = build_mtkan_model(MODEL_CONFIG_PATH)
            if os.path.exists(WEIGHTS_PATH):
                try:
                    model.load_weights(WEIGHTS_PATH)
                except Exception:
                    print("[TRAIN-MTKAN] Trọng số cũ không khớp kiến trúc mới. Huấn luyện mới hoàn toàn.")
                    
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                loss="mse",
                metrics=["mae"]
            )
            
            model.fit(
                X_train, y_train,
                epochs=5,
                batch_size=512,
                verbose=0
            )
            
            model.save_weights(WEIGHTS_PATH)
            print("[TRAIN-MTKAN SUCCESS] Đã huấn luyện lại và cập nhật trọng số MTKAN thành công!")
        except Exception as e:
            print(f"[TRAIN-MTKAN WARNING] Bỏ qua huấn luyện MTKAN do lỗi hoặc thay đổi kiến trúc: {e}. Hệ thống sẽ sử dụng Linear Regression làm phương án chính.")
    else:
        print("[TRAIN-MTKAN WARNING] Chưa đủ 100 dòng dữ liệu lịch sử để kích hoạt huấn luyện lại MTKAN.")

    print("[SUCCESS] Tiến trình huấn luyện hoàn thành tốt đẹp!")
    print("=" * 70)

if __name__ == "__main__":
    main()
