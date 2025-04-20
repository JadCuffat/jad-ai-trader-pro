import pandas as pd
import numpy as np
import requests
import json
import time
import os
from datetime import datetime
from binance.client import Client
from binance.enums import *
from binance_keys import API_KEY, API_SECRET
from telegram_config import BOT_TOKEN, CHAT_ID
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

# === Load and prepare data ===
data = pd.read_csv("training_data_multi_tf.csv")

required_cols = [
    'rsi_14_1h', 'macd_1h', 'macd_signal_1h', 'volume_spike_%_1h', 'price_above_ema_1h', 'atr_1h', 'momentum_1h', 'volatility_1h', 'normalized_volume_1h',
    'rsi_14_15m', 'macd_15m', 'macd_signal_15m', 'volume_spike_%_15m', 'price_above_ema_15m', 'atr_15m', 'momentum_15m', 'volatility_15m', 'normalized_volume_15m',
    'rsi_14_5m', 'macd_5m', 'macd_signal_5m', 'volume_spike_%_5m', 'price_above_ema_5m', 'atr_5m', 'momentum_5m', 'volatility_5m', 'normalized_volume_5m',
    'news_sentiment', 'target'
]

missing = [col for col in required_cols if col not in data.columns]
if missing:
    raise ValueError(f"Missing required columns in dataset: {missing}")

X = data.drop("target", axis=1)
y = data["target"]

# === Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# === Hyperparameter Tuning ===
params = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), params, cv=3, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

# === Evaluation ===
y_pred = grid.predict(X_test)
print(classification_report(y_test, y_pred))

# === Save Model ===
joblib.dump(grid.best_estimator_, "intraday_ai_model.joblib")
print("\n✅ Tuned model saved as 'intraday_ai_model.joblib'")
