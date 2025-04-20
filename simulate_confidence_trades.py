import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# Load profit-labeled data
df = pd.read_csv("ai_profit_labels.csv")
df.dropna(inplace=True)

# Features
features = [
    "rsi_14", "macd", "macd_signal", "sma_50", "ema_50",
    "volume_change_%", "dxy_close", "news_sentiment", "reddit_sentiment"
]

X = df[features]
y = df["target"]

# Train/test split
X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
    X, y, df, test_size=0.2, stratify=y, random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# Predict probabilities
proba = model.predict_proba(X_test)
preds = model.predict(X_test)

# Add predictions to test DataFrame
df_test = df_test.copy()
df_test["predicted"] = preds
df_test["confidence_buy"] = proba[:, list(model.classes_).index(1)]

# Filter: only high-confidence BUY predictions (confidence ≥ 0.50)
conf_threshold = 0.45
filtered = df_test[(df_test["predicted"] == 1) & (df_test["confidence_buy"] >= conf_threshold)]

# Simulate actual profit: use real next-day return
filtered["actual_profit_%"] = filtered["return_next_day_%"]

# Results
total_trades = len(filtered)
avg_profit = filtered["actual_profit_%"].mean()
win_rate = (filtered["actual_profit_%"] > 0).mean() * 100
total_return = filtered["actual_profit_%"].sum()

print("✅ Simulation Results (High-Confidence Buys Only):\n")
print(f"Total trades executed: {total_trades}")
print(f"Average daily return per trade: {avg_profit:.2f}%")
print(f"Win rate: {win_rate:.2f}%")
print(f"Total cumulative return: {total_return:.2f}%")

# Save trades to CSV
filtered.to_csv("simulated_trades.csv", index=False)
print("\n📁 Saved trades to simulated_trades.csv")
