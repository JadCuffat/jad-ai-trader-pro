```python
import pandas as pd
import numpy as np
import joblib
from binance.client import Client
from datetime import datetime, timedelta
from binance_keys import API_KEY, API_SECRET

# --- CONFIG ---
MODEL_PATH = 'intraday_ai_model.joblib'
START_DATE = '2025-03-01'
END_DATE   = '2025-04-18'
INITIAL_EQUITY = 10000.0
RISK_PER_TRADE = 0.02  # 2%
FEE_RATE = 0.001       # 0.1%
TIMEFRAME = '1h'
SYMBOL_LIMIT = 30

# --- INIT ---
client = Client(API_KEY, API_SECRET)
model  = joblib.load(MODEL_PATH)
classes = list(model.classes_)
equity = INITIAL_EQUITY
trade_log = []

# --- Fetch top USDT symbols excluding stablecoins ---
STABLECOIN_BASES = {"BUSD","USDC","TUSD","FDUSD","DAI","SUSDT","TUSDT"}
def get_top_usdt_symbols(limit=SYMBOL_LIMIT):
    tickers = client.get_ticker()
    usdt_pairs = []
    for t in tickers:
        symbol = t['symbol']
        base = symbol[:-4]
        if symbol.endswith('USDT') and base not in STABLECOIN_BASES:
            usdt_pairs.append(t)
    usdt_pairs.sort(key=lambda x: float(x.get('quoteVolume', 0)), reverse=True)
    return [t['symbol'] for t in usdt_pairs[:limit]]

SYMBOLS = get_top_usdt_symbols()

# --- FETCH HISTORICAL DATA ---
def fetch_klines(symbol, interval, start_str, end_str):
    klines = client.get_historical_klines(symbol, interval, start_str, end_str)
    df = pd.DataFrame(klines, columns=[
        'time','open','high','low','close','volume',
        'close_time','qav','trades','tbv','tqv','ignore'
    ])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
    return df.set_index('time')

# --- ADD INDICATORS ---
def add_indicators(df):
    df = df.copy()
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    df['rsi_14'] = 100 - (100/(1 + gain/loss))
    ema20 = df['close'].ewm(span=20).mean()
    df['price_above_ema'] = (df['close'] > ema20).astype(int)
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    vol_sma8 = df['volume'].rolling(8).mean()
    df['volume_spike_%'] = (df['volume'] - vol_sma8) / vol_sma8 * 100
    return df.dropna()

# --- RUN BACKTEST ---
for sym in SYMBOLS:
    print(f"Backtesting {sym}...")
    df = fetch_klines(sym, TIMEFRAME, START_DATE, END_DATE)
    df = add_indicators(df)
    for i in range(len(df)-1):
        row = df.iloc[i]
        features = pd.DataFrame([{ 
            'rsi_14': row['rsi_14'], 
            'price_above_ema': row['price_above_ema'],
            'macd': row['macd'], 
            'macd_signal': row['macd_signal'],
            'volume_spike_%': row['volume_spike_%']
        }])
        proba = model.predict_proba(features)[0]
        buy_conf  = proba[classes.index(1)] * 100
        sell_conf = proba[classes.index(-1)] * 100
        price = row['close']
        # Entry on next bar
        if buy_conf >= 75 and equity > 0:
            risk = equity * RISK_PER_TRADE
            qty = risk / price
            fee = price * qty * FEE_RATE
            entry_price = price
            exit_price = df['close'].iloc[i+1]
            pnl = (exit_price - entry_price) * qty - fee*2
            equity += pnl
            trade_log.append({
                'symbol': sym,
                'side': 'LONG',
                'entry': entry_price,
                'exit': exit_price,
                'pnl': pnl,
                'time': df.index[i]
            })
        elif sell_conf >= 75 and equity > 0:
            risk = equity * RISK_PER_TRADE
            qty = risk / price
            fee = price * qty * FEE_RATE
            entry_price = price
            exit_price = df['close'].iloc[i+1]
            pnl = (entry_price - exit_price) * qty - fee*2
            equity += pnl
            trade_log.append({
                'symbol': sym,
                'side': 'SHORT',
                'entry': entry_price,
                'exit': exit_price,
                'pnl': pnl,
                'time': df.index[i]
            })

# --- RESULTS ---
trades = pd.DataFrame(trade_log)
print(trades)
print(f"Final equity: ${equity:.2f}")
trades['date'] = trades['time'].dt.date
print(trades.groupby('date')['pnl'].sum().reset_index())
```
