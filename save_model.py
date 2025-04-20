import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load your training dataset
df = pd.read_csv("intraday_dataset.csv")
df.dropna(inplace=True)

features = [
    "rsi_14", "macd", "macd_signal", "ema_20", "volume_spike_%"
]
X = df[features]
y = df["pump_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=250,
    max_depth=12,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

# Save to file
joblib.dump(model, "intraday_ai_model.joblib")
print("✅ Model saved as intraday_ai_model.joblib")
