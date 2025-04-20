import pandas as pd
import os

symbols = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "DOTUSDT", "TONUSDT",
    "DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "WIFUSDT"
]

def label_pump_trades(df):
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["rsi_14"] = pd.to_numeric(df["rsi_14"], errors="coerce")
    df["macd"] = pd.to_numeric(df["macd"], errors="coerce")
    df["macd_signal"] = pd.to_numeric(df["macd_signal"], errors="coerce")
    df["ema_20"] = pd.to_numeric(df["ema_20"], errors="coerce")
    df["volume_spike_%"] = pd.to_numeric(df["volume_spike_%"], errors="coerce")

    signals = []

    for i in range(len(df)):
        row = df.iloc[i]

        # Entry condition
        if (
            row["rsi_14"] > 50 and
            row["macd"] > row["macd_signal"] and
            row["volume_spike_%"] > 30 and
            row["close"] > row["ema_20"]
        ):
            signals.append(1)

        # Exit condition
        elif (
            row["rsi_14"] < 60 or
            row["macd"] < row["macd_signal"] or
            row["volume_spike_%"] < 10 or
            row["close"] < row["ema_20"]
        ):
            signals.append(-1)
        else:
            signals.append(0)

    df["pump_label"] = signals
    return df

# Process each coin
for symbol in symbols:
    file = f"{symbol}_1h_indicators.csv"
    if os.path.exists(file):
        df = pd.read_csv(file, parse_dates=["timestamp"])
        df = label_pump_trades(df)
        df.to_csv(f"{symbol}_1h_labeled.csv", index=False)
        print(f"✅ Labeled {symbol} for pump signals.")
    else:
        print(f"❌ File not found: {file}")
