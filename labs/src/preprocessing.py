import pandas as pd
import numpy as np

# Load data
df_train = pd.read_csv(
    "/Users/jack/msds_coursework/mlops/labs/data/mobile_price/train.csv"
)
df_test = pd.read_csv(
    "/Users/jack/msds_coursework/mlops/labs/data/mobile_price/test.csv"
)

# Preprocess data
target_col = "price_range"
feature_cols = ["ram", "battery_power", "px_height", "px_width"]

train_data = df_train[feature_cols + [target_col]]
test_data = df_test[feature_cols]

# Save new data
train_data.to_csv("labs/data/processed_train_data.csv", index=False)
test_data.to_csv("labs/data/processed_test_data.csv", index=False)
