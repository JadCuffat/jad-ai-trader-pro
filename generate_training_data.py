import pandas as pd
import numpy as np
import requests
import json
import time
import os
from datetime import datetime, timedelta
from binance.client import Client
from binance.enums import *
from binance_keys import API_KEY, API_SECRET

client = Client(API_KEY, API_SECRET)

# === Fetch top 30 USDT pairs excluding stablecoins ===
def get_top_usdt_symbols(limit=30):
    stablecoin_bases = {"USDT", "BUSD", "USDC", "TUSD", "FDUSD", "DAI"}
    tickers = client.get_ticker()
    usdt_pairs = []
    for t in tickers:
        symbol = t["symbol"]
        if symbol.endswith("USDT"):
            base = symbol[:-4]
            if base not in stablecoin_bases:
                usdt_pairs.append(t)
    usdt_pairs.sort(key=lambda x: float(x.get("quoteVolume", 0)), reverse=True)
    return [t["symbol"] for t in usdt_pairs[:limit]]

# === News sentiment via CryptoPanic ===
def get_news_sentiment(symbol):
    try:
        resp = requests.get(
            f"https://cryptopanic.com/api/v1/posts/?auth_token=YOUR_CRYPTOPANIC_API_KEY&currencies={symbol[:3].lower()}&public=true"
        )
        data = resp.json()
        score = 0
        count = 0
        for item in data.get("results", []):
            s = item.get('sentiment')
            if s == 'positive': score += 1
            if s == 'negative': score -= 1
            if s in ('positive','negative'): count += 1
        return score / count if count else 0
    except Exception:
        return 0

# === Compute indicators ===
def add_indicators(df):
    # ensure numeric
    for col in ['open','high','low','close','volume']:
        df[col] = pd.to_numeric(df[col])

    # RSI 14
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # EMA20 and price above EMA
    ema20 = df['close'].ewm(span=20, adjust=False).mean()
    df['price_above_ema'] = (df['close'] > ema20).astype(int)

    # MACD and signal
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # Volume spike
    df['volume_sma_8'] = df['volume'].rolling(8).mean()
    df['volume_spike_%'] = ((df['volume'] - df['volume_sma_8']) / df['volume_sma_8']) * 100

    # ATR
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()

    # Momentum
    df['momentum'] = df['close'] - df['close'].shift(5)

    # Volatility
    df['volatility'] = df['close'].rolling(14).std()

    # Normalized volume
    df['normalized_volume'] = df['volume'] / df['volume'].rolling(14).mean()

    return df.dropna()

# === Fetch OHLCV ===
def fetch_ohlcv(symbol, interval):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=500"
    resp = requests.get(url)
    data = resp.json()
    cols = ['time','open','high','low','close','volume','close_time','qav','trades','tbbav','tbqav','ignore']
    df = pd.DataFrame(data, columns=cols)
    return df[['time','open','high','low','close','volume']]

# === Build and save dataset ===
def build_dataset():
    symbols = get_top_usdt_symbols(limit=30)
    rows = []
    for idx, sym in enumerate(symbols, 1):
        print(f"📊 Processing {sym} ({idx}/{len(symbols)})...")
        try:
            df1h = add_indicators(fetch_ohlcv(sym, '1h'))
            df15m = add_indicators(fetch_ohlcv(sym, '15m'))
            df5m  = add_indicators(fetch_ohlcv(sym, '5m'))
            for i in range(len(df5m)-1):
                feat = {
                    # 1h features
                    'rsi_14_1h': df1h.iloc[i]['rsi_14'],
                    'macd_1h': df1h.iloc[i]['macd'],
                    'macd_signal_1h': df1h.iloc[i]['macd_signal'],
                    'volume_spike_%_1h': df1h.iloc[i]['volume_spike_%'],
                    'price_above_ema_1h': df1h.iloc[i]['price_above_ema'],
                    'atr_1h': df1h.iloc[i]['atr'],
                    'momentum_1h': df1h.iloc[i]['momentum'],
                    'volatility_1h': df1h.iloc[i]['volatility'],
                    'normalized_volume_1h': df1h.iloc[i]['normalized_volume'],
                    # 15m features
                    'rsi_14_15m': df15m.iloc[i]['rsi_14'],
                    'macd_15m': df15m.iloc[i]['macd'],
                    'macd_signal_15m': df15m.iloc[i]['macd_signal'],
                    'volume_spike_%_15m': df15m.iloc[i]['volume_spike_%'],
                    'price_above_ema_15m': df15m.iloc[i]['price_above_ema'],
                    'atr_15m': df15m.iloc[i]['atr'],
                    'momentum_15m': df15m.iloc[i]['momentum'],
                    'volatility_15m': df15m.iloc[i]['volatility'],
                    'normalized_volume_15m': df15m.iloc[i]['normalized_volume'],
                    # 5m features
                    'rsi_14_5m': df5m.iloc[i]['rsi_14'],
                    'macd_5m': df5m.iloc[i]['macd'],
                    'macd_signal_5m': df5m.iloc[i]['macd_signal'],
                    'volume_spike_%_5m': df5m.iloc[i]['volume_spike_%'],
                    'price_above_ema_5m': df5m.iloc[i]['price_above_ema'],
                    'atr_5m': df5m.iloc[i]['atr'],
                    'momentum_5m': df5m.iloc[i]['momentum'],
                    'volatility_5m': df5m.iloc[i]['volatility'],
                    'normalized_volume_5m': df5m.iloc[i]['normalized_volume'],
                    # sentiment
                    'news_sentiment': get_news_sentiment(sym)
                }
                # compute target based on next 5m close
                cur = df5m.iloc[i]['close']
                nxt = df5m.iloc[i+1]['close']
                change = (nxt - cur) / cur
                feat['target'] = 1 if change > 0.003 else (-1 if change < -0.003 else 0)
                rows.append(feat)
        except Exception as e:
            print(f"⚠️ Skipping {sym}: {e}")

    df = pd.DataFrame(rows)
    df.dropna(inplace=True)
    if df.empty:
        print("⚠️ No data collected.")
        return
    # balance classes
    m = df['target'].value_counts().min()
    df_bal = pd.concat([
        df[df.target==1].sample(m, random_state=42),
        df[df.target==-1].sample(m, random_state=42),
        df[df.target==0].sample(m, random_state=42)
    ])
    df_bal.to_csv('training_data_multi_tf.csv', index=False)
    print("✅ Balanced dataset saved as 'training_data_multi_tf.csv'")

if __name__ == '__main__':
    build_dataset()
