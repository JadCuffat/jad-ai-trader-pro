import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load and clean dataset
df = pd.read_csv("intraday_dataset.csv")
df.dropna(inplace=True)
df["close"] = pd.to_numeric(df["close"], errors="coerce")

# Define features and target
features = [
    "rsi_14", "macd", "macd_signal", "ema_20", "volume_spike_%"
]
X = df[features]
y = df["pump_label"]

# Train/test split
X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
    X, y, df, test_size=0.2, stratify=y, random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=250,
    max_depth=12,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

# Predict on test set
df_test = df_test.copy()
df_test["close"] = pd.to_numeric(df_test["close"], errors="coerce")
df_test["prediction"] = model.predict(X_test)
df_test["position"] = 0

# Simulate trading
in_position = False
entry_price = 0
pnl_list = []

for i in range(len(df_test)):
    pred = df_test.iloc[i]["prediction"]
    try:
        price = float(df_test["close"].iloc[i])
    except:
        continue

    if not in_position and pred == 1 and price > 0:
        entry_price = price
        df_test.at[df_test.index[i], "position"] = 1
        in_position = True
        print(f"[ENTRY] at {entry_price:.4f}")

    elif in_position and pred == -1 and price > 0:
        try:
            pnl = ((price - entry_price) / entry_price) * 100
            if abs(pnl) < 50:  # ignore unrealistic moves
                pnl_list.append(pnl)
                print(f"[EXIT] at {price:.4f} — PnL: {pnl:.2f}%")
        except:
            continue
        in_position = False
        df_test.at[df_test.index[i], "position"] = -1

# Results
total_trades = len(pnl_list)
average_return = sum(pnl_list) / total_trades if total_trades > 0 else 0
win_rate = len([x for x in pnl_list if x > 0]) / total_trades * 100 if total_trades > 0 else 0
cumulative_return = sum(pnl_list)

print("\n✅ Backtest Simulation Results:\n")
print(f"📈 Trades Executed: {total_trades}")
print(f"💰 Avg Return per Trade: {average_return:.2f}%")
print(f"✅ Win Rate: {win_rate:.2f}%")
print(f"📊 Total Cumulative Return: {cumulative_return:.2f}%")

df_test.to_csv("simulated_intraday_trades.csv", index=False)
print("\n📁 Saved results to simulated_intraday_trades.csv")
