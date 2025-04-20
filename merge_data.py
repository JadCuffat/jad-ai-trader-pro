import pandas as pd

# Load BTC price
btc = pd.read_csv("BTCUSDT_historical.csv", parse_dates=["timestamp"])
btc.rename(columns={
    "timestamp": "date",
    "open": "btc_open",
    "high": "btc_high",
    "low": "btc_low",
    "close": "btc_close",
    "volume": "btc_volume"
}, inplace=True)

# Load DXY
dxy = pd.read_csv("DXY_historical.csv", parse_dates=["Date"])
dxy.rename(columns={
    "Date": "date",
    "Open": "dxy_open",
    "High": "dxy_high",
    "Low": "dxy_low",
    "Close": "dxy_close",
    "Volume": "dxy_volume"
}, inplace=True)

# Load News Sentiment
news = pd.read_csv("crypto_news_sentiment.csv", parse_dates=["date"])

# Merge all
df = btc.merge(dxy, on="date", how="left")
df = df.merge(news, on="date", how="left")

# Fill missing sentiment with 0
df["news_sentiment"].fillna(0, inplace=True)

# Save merged result
df.to_csv("merged_dataset.csv", index=False)
print("✅ Merged dataset saved as merged_dataset.csv")
