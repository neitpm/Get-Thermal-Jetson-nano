import pandas as pd
from src.predict import predict_all_windows

def main():
    input_csv = "data/39-43.csv"
    output_csv = "data/predictions_all_windows.csv"

    df = pd.read_csv(input_csv)

    pred_df = predict_all_windows(df)

    print("Kết quả dự đoán:")
    print(pred_df.head())

    pred_df.to_csv(output_csv, index=False)
    print(f"Đã lưu kết quả vào: {output_csv}")

if __name__ == "__main__":
    main()
