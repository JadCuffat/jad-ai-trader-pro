import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load new profit-labeled dataset
df = pd.read_csv("ai_profit_labels.csv")
df.dropna(inplace=True)

# Define features for decision making
features = [
    "rsi_14", "macd", "macd_signal", "sma_50", "ema_50",
    "volume_change_%", "dxy_close", "news_sentiment", "reddit_sentiment"
]

X = df[features]
y = df["target"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train the model with class balancing
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate performance
y_pred = model.predict(X_test)
print("✅ Profit-oriented AI trained!\n")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
