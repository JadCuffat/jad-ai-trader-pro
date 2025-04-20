import pandas as pd
import os

# List of your coin files
symbols = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "DOTUSDT", "TONUSDT",
    "DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "WIFUSDT"
]

def compute_indicators(df):
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    # RSI (14)
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD and Signal (12, 26, 9)
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # SMA and EMA (50)
    df["sma_50"] = df["close"].rolling(window=50).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

    # Volume change (%)
    df["volume_change_%"] = df["volume"].pct_change() * 100

    return df

# Process each file
for symbol in symbols:
    file = f"{symbol}_historical.csv"
    if os.path.exists(file):
        df = pd.read_csv(file, parse_dates=["timestamp"])
        df = compute_indicators(df)
        df.to_csv(f"{symbol}_indicators.csv", index=False)
        print(f"✅ {symbol} indicators saved.")
    else:
        print(f"❌ File not found: {file}")
