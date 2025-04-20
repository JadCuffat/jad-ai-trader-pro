import pandas as pd
import os

symbols = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "DOTUSDT", "TONUSDT",
    "DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "WIFUSDT"
]

combined = []

for symbol in symbols:
    file = f"{symbol}_1h_labeled.csv"
    if os.path.exists(file):
        df = pd.read_csv(file, parse_dates=["timestamp"])
        df["coin"] = symbol  # add coin tag
        combined.append(df)
        print(f"✅ Loaded: {symbol}")
    else:
        print(f"❌ Missing: {file}")

# Combine all into one DataFrame
full_df = pd.concat(combined, ignore_index=True)
full_df.dropna(inplace=True)  # clean NaNs
full_df.to_csv("intraday_dataset.csv", index=False)
print("\n📁 Saved combined dataset as intraday_dataset.csv")
