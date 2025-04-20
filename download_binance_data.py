import requests
import pandas as pd
import time

BASE_URL = "https://api.binance.com/api/v3/klines"

def download_binance_ohlcv(symbol, interval="1d", limit=1000):
    url = f"{BASE_URL}?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed for {symbol}")
        return None

    data = response.json()
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df.to_csv(f"{symbol}_historical.csv", index=False)
    print(f"✅ Saved {symbol}_historical.csv")

# Your selected coins
symbols = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "DOTUSDT", "TONUSDT",
    "DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "WIFUSDT"
]

# Download all
for symbol in symbols:
    download_binance_ohlcv(symbol, interval="1d")
    time.sleep(0.5)  # avoid rate limiting
