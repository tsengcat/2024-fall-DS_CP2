import pandas as pd
import os
import numpy as np
import zipfile
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# 定義數據路徑和文件名稱
train_folder = "DS_Dataset"
output_file = "upload.csv"
zip_file = "upload.zip"

# 初始化字典來存儲每個裝置的模型和訓練數據
models = {}
train_data = {}

# 讀取所有訓練文件並為每個裝置訓練模型
for location in range(1, 18):
    train_file = os.path.join(train_folder, f"L{location}_train.csv")
    train_df = pd.read_csv(train_file)

    # 處理日期時間
    train_df["DateTime"] = pd.to_datetime(train_df["DateTime"])
    train_df["Hour"] = train_df["DateTime"].dt.hour
    train_df["Minute"] = train_df["DateTime"].dt.minute
    train_df["DayOfWeek"] = train_df["DateTime"].dt.dayofweek

    # 特徵與標籤
    features = ["WindSpeed(m/s)", "Pressure(hpa)", "Temperature(°C)", "Humidity(%)", "Sunlight(Lux)", "Hour", "Minute", "DayOfWeek"]
    X = train_df[features]
    y = train_df["Power(mW)"]

    # 切分數據集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 訓練模型
    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    # 驗證模型
    val_predictions = model.predict(X_val)
    print(f"Location {location}: Validation MAE: {np.mean(np.abs(val_predictions - y_val))}")

    # 保存模型和訓練數據
    models[location] = model
    train_data[location] = train_df

# 讀取 upload.csv
upload_path = os.path.join(train_folder, "upload.csv")
upload_df = pd.read_csv(upload_path, header=0, names=["序號", "答案"])

# 確保序號列為字符串
upload_df["序號"] = upload_df["序號"].astype(str)

# 提取時間和裝置代碼特徵
upload_df["DateTime"] = pd.to_datetime(upload_df["序號"].str[:12], format="%Y%m%d%H%M")
upload_df["LocationCode"] = upload_df["序號"].str[-2:].astype(int)
upload_df["Hour"] = upload_df["DateTime"].dt.hour
upload_df["Minute"] = upload_df["DateTime"].dt.minute
upload_df["DayOfWeek"] = upload_df["DateTime"].dt.dayofweek

# 準備特徵列表
features = ["WindSpeed(m/s)", "Pressure(hpa)", "Temperature(°C)", "Humidity(%)", "Sunlight(Lux)", "Hour", "Minute", "DayOfWeek"]

# 預測每一行的答案
predictions = []
for _, row in upload_df.iterrows():
    location = row["LocationCode"]
    model = models[location]  # 使用對應的裝置模型

    # 查找對應裝置的訓練數據
    location_data = train_data[location]
    
    # 找到最近的特徵值
    closest_row = location_data.iloc[(location_data["DateTime"] - row["DateTime"]).abs().argsort()[:1]]
    X_pred = closest_row[features]

    # 預測
    prediction = model.predict(X_pred)[0]
    predictions.append(max(0, round(prediction, 2)))  # 確保非負，格式化到小數點後兩位

# 保存預測結果
upload_df["答案"] = predictions
upload_df[["序號", "答案"]].to_csv(output_file, index=False, header=True, encoding='utf-8-sig')

# 壓縮為 zip 文件
with zipfile.ZipFile(zip_file, 'w') as zf:
    zf.write(output_file)
