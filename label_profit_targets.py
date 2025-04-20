import pandas as pd

# Load dataset
df = pd.read_csv("ai_master_dataset.csv", parse_dates=["date"])
df["close"] = pd.to_numeric(df["close"], errors="coerce")
df.sort_values(["coin", "date"], inplace=True)

# Compute next day return %
df["next_close"] = df.groupby("coin")["close"].shift(-1)
df["return_next_day_%"] = (df["next_close"] - df["close"]) / df["close"] * 100

# New label logic
def smart_label(pct):
    if pct >= 2:
        return 1   # Good trade entry
    elif pct <= -1.5:
        return -1  # Avoid or exit
    else:
        return 0   # Hold / Ignore

df["target"] = df["return_next_day_%"].apply(smart_label)

# Drop helper columns
df.drop(columns=["next_close"], inplace=True)

# Save new file
df.to_csv("ai_profit_labels.csv", index=False)
print("✅ ai_profit_labels.csv saved with 2% profit target logic")
