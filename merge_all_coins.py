import pandas as pd
import os

# List of coins
symbols = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "DOTUSDT", "TONUSDT",
    "DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "WIFUSDT"
]

# Load macro and sentiment
dxy = pd.read_csv("DXY_historical.csv", parse_dates=["Date"])
dxy.rename(columns={"Date": "date", "Close": "dxy_close"}, inplace=True)
dxy = dxy[["date", "dxy_close"]]

news = pd.read_csv("crypto_news_sentiment.csv", parse_dates=["date"])
reddit = pd.read_csv("reddit_sentiment.csv", parse_dates=["date"])

# Master list to collect everything
all_data = []

# Process each coin
for symbol in symbols:
    file = f"{symbol}_indicators.csv"
    if os.path.exists(file):
        df = pd.read_csv(file, parse_dates=["timestamp"])
        df.rename(columns={"timestamp": "date"}, inplace=True)

        # Merge macro + sentiment
        df = df.merge(dxy, on="date", how="left")
        df = df.merge(news, on="date", how="left")
        df = df.merge(reddit, on="date", how="left")

        df["coin"] = symbol  # Add coin tag
        df["news_sentiment"].fillna(0, inplace=True)
        df["reddit_sentiment"].fillna(0, inplace=True)

        all_data.append(df)
        print(f"✅ Merged {symbol}")
    else:
        print(f"❌ Missing file: {file}")

# Concatenate all coins
final = pd.concat(all_data, ignore_index=True)
final.to_csv("ai_master_dataset.csv", index=False)
print("✅ All coins merged into ai_master_dataset.csv")
