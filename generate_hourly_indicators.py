import pandas as pd
import os

# Your 13 coins
symbols = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "DOTUSDT", "TONUSDT",
    "DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "WIFUSDT"
]

def compute_indicators(df):
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # EMA
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()

    # Volume spike
    df["volume_sma_8"] = df["volume"].rolling(window=8).mean()
    df["volume_spike_%"] = ((df["volume"] - df["volume_sma_8"]) / df["volume_sma_8"]) * 100

    return df

# Process all 13 coins
for symbol in symbols:
    file = f"{symbol}_1h.csv"
    if os.path.exists(file):
        df = pd.read_csv(file, parse_dates=["timestamp"])
        df = compute_indicators(df)
        df.to_csv(f"{symbol}_1h_indicators.csv", index=False)
        print(f"✅ {symbol} indicators added.")
    else:
        print(f"❌ File not found: {file}")
