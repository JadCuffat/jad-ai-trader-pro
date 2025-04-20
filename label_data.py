import pandas as pd

# Load dataset
df = pd.read_csv("ai_master_dataset.csv", parse_dates=["date"])

# Ensure numeric
df["close"] = pd.to_numeric(df["close"], errors="coerce")

# Sort by coin + date
df.sort_values(["coin", "date"], inplace=True)

# Next day close % return
df["next_close"] = df.groupby("coin")["close"].shift(-1)
df["return_next_day_%"] = (df["next_close"] - df["close"]) / df["close"] * 100

# Basic label logic
def get_label(pct):
    if pct > 2:
        return 1   # Buy
    elif pct < -2:
        return -1  # Sell
    else:
        return 0   # Hold

df["target"] = df["return_next_day_%"].apply(get_label)

# ✅ Prevent consecutive SELL (-1) spamming
df["prev_target"] = df.groupby("coin")["target"].shift(1)
df.loc[(df["target"] == -1) & (df["prev_target"] == -1), "target"] = 0
df.drop(columns=["next_close", "prev_target"], inplace=True)

# Save it
df.to_csv("ai_labeled_dataset.csv", index=False)
print("✅ ai_labeled_dataset.csv saved with smart targets")
