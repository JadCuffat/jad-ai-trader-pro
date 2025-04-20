import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
from binance.client import Client
from binance.exceptions import BinanceAPIException
from binance_keys import API_KEY, API_SECRET
from telegram_config import BOT_TOKEN, CHAT_ID
import joblib

# Binance + model setup
client = Client(API_KEY, API_SECRET)
model = joblib.load("intraday_ai_model.joblib")
CONFIDENCE_THRESHOLD = 75  # now at 75%

# 30 top USDT coins, excluding stablecoins
def get_top_usdt_symbols(limit=30):
    stablecoin_kw = {"BUSD","USDC","TUSD","FDUSD","DAI","SUSDT","TUSDT"}
    tickers = client.get_ticker()
    usdt = [t for t in tickers
            if t["symbol"].endswith("USDT")
            and all(st not in t["symbol"][:-4] for st in stablecoin_kw)]
    usdt.sort(key=lambda x: float(x.get("quoteVolume",0)), reverse=True)
    return [t["symbol"] for t in usdt[:limit]]

# Telegram alert helper
def send_telegram(msg):
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": msg}
        )
    except:
        pass

# CryptoPanic sentiment (replace YOUR_CRYPTOPANIC_API_KEY)
def get_news_sentiment(symbol):
    try:
        r = requests.get(
          f"https://cryptopanic.com/api/v1/posts/?auth_token=YOUR_CRYPTOPANIC_API_KEY"
          f"&currencies={symbol[:3].lower()}&public=true"
        ).json()
        s = 0; n=0
        for item in r.get("results",[]):
            if item.get("sentiment")=="positive": s+=1
            elif item.get("sentiment")=="negative": s-=1
            n+=1
        return s/n if n else 0
    except:
        return 0

# Indicator builder with suffix (_1h, _15m, _5m)
def add_indicators(df, suffix=""):
    df = df.copy()
    df['rsi_14'+suffix] = (df['close'].diff().clip(lower=0)
        .rolling(14).mean()
      / df['close'].diff().abs().rolling(14).mean() * 100)
    ema = df['close'].ewm(span=20,adjust=False).mean()
    df['price_above_ema'+suffix] = (df['close']>ema).astype(int)
    df['macd'+suffix] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['macd_signal'+suffix] = df['macd'+suffix].ewm(span=9).mean()
    df['volume_sma_8'+suffix] = df['volume'].rolling(8).mean()
    df['volume_spike_%'+suffix] = ((df['volume'] - df['volume_sma_8'+suffix])
                                  / df['volume_sma_8'+suffix])*100
    df['atr'+suffix] = (df['high']-df['low']).rolling(14).mean()
    df['momentum'+suffix] = df['close'] - df['close'].shift(5)
    df['volatility'+suffix] = df['close'].rolling(14).std()
    df['normalized_volume'+suffix] = df['volume']/df['volume'].rolling(14).mean()
    return df.dropna()

# OHLCV fetcher
def fetch_ohlcv(symbol, interval):
    kl = client.get_klines(symbol=symbol, interval=interval, limit=100)
    df = pd.DataFrame(kl,columns=[
        "time","open","high","low","close","volume","ct","qav","trades","tbv","tqv","ignore"
    ])
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
    df["time"] = pd.to_datetime(df["time"],unit="ms")
    return df

# Main loop
def run_signals():
    print(f"\n📡 Jad’s AI Signal Engine — {datetime.now():%Y‑%m‑%d %H:%M}\n")
    for sym in get_top_usdt_symbols():
        print(f"🔎 Analyzing {sym}...")
        try:
            df1 = add_indicators(fetch_ohlcv(sym,"1h"), "_1h")
            df15= add_indicators(fetch_ohlcv(sym,"15m"), "_15m")
            df5 = add_indicators(fetch_ohlcv(sym,"5m"), "_5m")
            sent = get_news_sentiment(sym)
            if df1.empty or df15.empty or df5.empty:
                continue

            feat = {
                # 1h
                "rsi_14_1h": df1.iloc[-1]["rsi_14_1h"],
                "macd_1h": df1.iloc[-1]["macd_1h"],
                "macd_signal_1h": df1.iloc[-1]["macd_signal_1h"],
                "volume_spike_%_1h": df1.iloc[-1]["volume_spike_%_1h"],
                "price_above_ema_1h": df1.iloc[-1]["price_above_ema_1h"],
                "atr_1h": df1.iloc[-1]["atr_1h"],
                "momentum_1h": df1.iloc[-1]["momentum_1h"],
                "volatility_1h": df1.iloc[-1]["volatility_1h"],
                "normalized_volume_1h": df1.iloc[-1]["normalized_volume_1h"],
                # 15m
                "rsi_14_15m": df15.iloc[-1]["rsi_14_15m"],
                "macd_15m": df15.iloc[-1]["macd_15m"],
                "macd_signal_15m": df15.iloc[-1]["macd_signal_15m"],
                "volume_spike_%_15m": df15.iloc[-1]["volume_spike_%_15m"],
                "price_above_ema_15m": df15.iloc[-1]["price_above_ema_15m"],
                "atr_15m": df15.iloc[-1]["atr_15m"],
                "momentum_15m": df15.iloc[-1]["momentum_15m"],
                "volatility_15m": df15.iloc[-1]["volatility_15m"],
                "normalized_volume_15m": df15.iloc[-1]["normalized_volume_15m"],
                # 5m
                "rsi_14_5m": df5.iloc[-1]["rsi_14_5m"],
                "macd_5m": df5.iloc[-1]["macd_5m"],
                "macd_signal_5m": df5.iloc[-1]["macd_signal_5m"],
                "volume_spike_%_5m": df5.iloc[-1]["volume_spike_%_5m"],
                "price_above_ema_5m": df5.iloc[-1]["price_above_ema_5m"],
                "atr_5m": df5.iloc[-1]["atr_5m"],
                "momentum_5m": df5.iloc[-1]["momentum_5m"],
                "volatility_5m": df5.iloc[-1]["volatility_5m"],
                "normalized_volume_5m": df5.iloc[-1]["normalized_volume_5m"],
                # sentiment
                "news_sentiment": sent
            }
            X = pd.DataFrame([feat])
            missing = [c for c in model.feature_names_in_ if c not in X.columns]
            if missing:
                print(f"⚠️ Skipping {sym}: Missing {missing}")
                continue

            p = model.predict_proba(X[model.feature_names_in_])[0]
            buy_conf, sell_conf = p[2]*100, p[0]*100

            # BUY
            if buy_conf >= CONFIDENCE_THRESHOLD:
                try:
                    price = float(client.get_symbol_ticker(symbol=sym)['price'])
                    amt = 20
                    prec = 6
                    qty = round(amt/price,prec)
                    ord = client.order_market_buy(symbol=sym,quantity=qty)
                    p_exec = float(ord['fills'][0]['price'])
                    q_exec = ord['executedQty']
                    msg = f"✅ BUY {sym} — Qty:{q_exec} @ {p_exec:.4f} — Conf:{buy_conf:.1f}%"
                    print(msg); send_telegram(msg)
                except BinanceAPIException as e:
                    print(f"❌ Buy error for {sym}: {e.message}")

            # SELL
            elif sell_conf >= CONFIDENCE_THRESHOLD:
                try:
                    bal = float(client.get_asset_balance(asset=sym.replace("USDT",""))['free'])
                    prec=6; qty=round(bal,prec)
                    ord = client.order_market_sell(symbol=sym,quantity=qty)
                    p_exec = float(ord['fills'][0]['price'])
                    q_exec = ord['executedQty']
                    msg = f"📤 SELL {sym} — Qty:{q_exec} @ {p_exec:.4f} — Conf:{sell_conf:.1f}%"
                    print(msg); send_telegram(msg)
                except BinanceAPIException as e:
                    print(f"❌ Sell error for {sym}: {e.message}")

            else:
                print(f"[{sym}] → HOLD (Buy:{buy_conf:.1f}%, Sell:{sell_conf:.1f}%)")

        except Exception as ex:
            print(f"❌ Prediction error for {sym}: {ex}")

if __name__ == "__main__":
    # initial ping
    send_telegram("🌀 Jad’s AI Bot launched successfully and awaiting next signal cycle...")
    while True:
        run_signals()
        time.sleep(300)
