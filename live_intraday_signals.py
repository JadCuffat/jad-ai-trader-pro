from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException
from binance_keys import API_KEY, API_SECRET
client = Client(API_KEY, API_SECRET)

from telegram_config import BOT_TOKEN, CHAT_ID

import requests
import pandas as pd
import joblib
import time
import csv
import os
import json
from datetime import datetime

print("🌀 Jad’s AI Bot launched successfully and awaiting next signal cycle...")

model = joblib.load("intraday_ai_model.joblib")

core_symbols = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "DOTUSDT", "TONUSDT",
    "DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "WIFUSDT"
]

def get_top_usdt_pairs(limit=15):
    try:
        tickers = client.get_ticker()
        usdt_pairs = [t for t in tickers if t['symbol'].endswith("USDT") and not t['symbol'].startswith("LD")]
        sorted_pairs = sorted(usdt_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)
        return [pair['symbol'] for pair in sorted_pairs if pair['symbol'] not in core_symbols][:limit]
    except Exception as e:
        print("❌ Failed to fetch USDT pairs:", e)
        return []

positions_file = "positions.json"

# === Telegram Alert ===
def send_telegram(message):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message}
        requests.post(url, data=payload)
    except Exception as e:
        print("❌ Telegram error:", e)

# === Trade Logger ===
def log_trade(symbol, side, confidence, price, quantity, tag="core"):
    filename = "executed_trades.csv"
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "Symbol", "Side", "Confidence", "Price", "Quantity", "Tag"])
        writer.writerow([datetime.now(), symbol, side, f"{confidence:.2f}%", price, quantity, tag])

# === Daily PnL Summary ===
def log_daily_pnl():
    try:
        df = pd.read_csv("executed_trades.csv", on_bad_lines='skip')
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["Date"] = df["Timestamp"].dt.date
        df["Price"] = pd.to_numeric(df["Price"])
        df["Quantity"] = pd.to_numeric(df["Quantity"])

        pnl_summary = []
        for date, group in df.groupby("Date"):
            profit = 0
            buys = {}
            for _, row in group.iterrows():
                symbol = row["Symbol"]
                if row["Side"] == "BUY":
                    buys[symbol] = (row["Price"], row["Quantity"])
                elif row["Side"] == "SELL" and symbol in buys:
                    buy_price, qty = buys[symbol]
                    sell_price = row["Price"]
                    pnl = (sell_price - buy_price) * qty
                    profit += pnl
                    del buys[symbol]
            pnl_summary.append([date, profit])

        with open("daily_pnl_summary.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Date", "Realized PnL (USD)"])
            writer.writerows(pnl_summary)
    except Exception as e:
        print("❌ PnL log error:", e)

# === Positions Tracker ===
def load_positions():
    if os.path.exists(positions_file):
        with open(positions_file, "r") as f:
            return json.load(f)
    return {}

def save_positions(positions):
    with open(positions_file, "w") as f:
        json.dump(positions, f, indent=2)

# === Precision Helper ===
def get_quantity_precision(symbol):
    try:
        info = client.get_symbol_info(symbol)
        for f in info["filters"]:
            if f["filterType"] == "LOT_SIZE":
                step = float(f["stepSize"])
                return len(str(step).split(".")[-1].rstrip("0"))
    except:
        return 4
    return 4

# === Liquidity Checkers ===
def is_liquid(symbol, min_quote_volume=5000000, min_bid_ask_ratio=0.95):
    try:
        book = client.get_order_book(symbol=symbol, limit=5)
        ticker = client.get_ticker(symbol=symbol)
        bid = float(book['bids'][0][0])
        ask = float(book['asks'][0][0])
        quote_volume = float(ticker['quoteVolume'])
        spread_ratio = bid / ask if ask > 0 else 0
        return quote_volume >= min_quote_volume and spread_ratio >= min_bid_ask_ratio
    except:
        return False

def can_exit_liquidly(symbol, quantity):
    try:
        order_book = client.get_order_book(symbol=symbol, limit=10)
        total_bid_value = sum(float(bid[0]) * float(bid[1]) for bid in order_book['bids'])
        price = float(client.get_symbol_ticker(symbol=symbol)["price"])
        return total_bid_value >= price * quantity * 0.95
    except:
        return False

# === Trade Execution ===
def execute_trade(symbol, side, usdt_amount=None, fixed_quantity=None):
    try:
        price = float(client.get_symbol_ticker(symbol=symbol)["price"])
        precision = get_quantity_precision(symbol)

        if fixed_quantity is not None:
            quantity = round(fixed_quantity, precision)
        else:
            quantity = round(usdt_amount / price, precision)

        if side == "SELL":
            asset = symbol.replace("USDT", "")
            balance = float(client.get_asset_balance(asset=asset)["free"])
            quantity = min(quantity, balance)
            quantity = round(quantity - 10**-precision, precision)
            if quantity <= 0:
                print(f"⚠️ Skipping SELL: not enough balance for {symbol} (Have: {balance}, Need: {quantity})")
                return None, None

        order = client.create_order(
            symbol=symbol,
            side=SIDE_BUY if side == "BUY" else SIDE_SELL,
            type=ORDER_TYPE_MARKET,
            quantity=quantity
        )

        print(f"✅ {side} {symbol}: Qty={quantity}, Price={price:.4f}")
        return price, quantity

    except BinanceAPIException as e:
        print(f"❌ Binance error: {e}")
    except Exception as ex:
        print(f"❌ General error: {ex}")
    return None, None

# === Fetch Candles ===
def fetch_ohlcv(symbol):
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=50"
        response = requests.get(url)
        if response.status_code != 200:
            return None
        data = response.json()
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "qav", "trades", "tbbav", "tbqav", "ignore"
        ])
        df["close"] = pd.to_numeric(df["close"])
        df["volume"] = pd.to_numeric(df["volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df[["timestamp", "close", "volume"]]
    except:
        return None

# === Add Indicators ===
def add_indicators(df):
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["volume_sma_8"] = df["volume"].rolling(window=8).mean()
    df["volume_spike_%"] = ((df["volume"] - df["volume_sma_8"]) / df["volume_sma_8"]) * 100
    return df

# === Run Prediction ===
def run_prediction():
    print("\n📡 Jad’s AI Signal Engine — Status\n")
    positions = load_positions()
    no_trades = True
    all_symbols = core_symbols + get_top_usdt_pairs()

    for symbol in all_symbols:
        df = fetch_ohlcv(symbol)
        if df is None or len(df) < 30:
            continue
        df = add_indicators(df).dropna()
        latest = df.iloc[-1]
        features = pd.DataFrame([{ 
            "rsi_14": latest["rsi_14"],
            "macd": latest["macd"],
            "macd_signal": latest["macd_signal"],
            "ema_20": latest["ema_20"],
            "volume_spike_%": latest["volume_spike_%"]
        }])

        proba = model.predict_proba(features)[0]
        buy_conf, sell_conf = proba[1] * 100, proba[0] * 100

        if buy_conf >= 80 and symbol not in positions and is_liquid(symbol):
            price, qty = execute_trade(symbol, "BUY", usdt_amount=20)
            if price and qty:
                positions[symbol] = {
                    "buy_price": price,
                    "quantity": qty,
                    "timestamp": str(datetime.now())
                }
                send_telegram(f"✅ BUY {symbol} @ {price:.4f} | Qty: {qty} | Conf: {buy_conf:.1f}%")
                log_trade(symbol, "BUY", buy_conf, price, qty)
                no_trades = False

        elif sell_conf >= 80 and symbol in positions:
            entry = positions[symbol]
            if can_exit_liquidly(symbol, entry["quantity"]):
                price, qty = execute_trade(symbol, "SELL", fixed_quantity=entry["quantity"])
                if price and qty:
                    pnl = (price - entry["buy_price"]) * qty
                    send_telegram(f"📤 SELL {symbol} @ {price:.4f} | PnL: ${pnl:.2f}")
                    log_trade(symbol, "SELL", sell_conf, price, qty)
                    del positions[symbol]
                    no_trades = False

        print(f"[{symbol}] → {'ENTER' if buy_conf >= 80 else 'EXIT'} (Buy: {buy_conf:.1f}%, Sell: {sell_conf:.1f}%)")

    save_positions(positions)
    log_daily_pnl()

    if no_trades:
        send_telegram("📭 No trades this cycle. All signals HOLD or low confidence.")

# === Main Loop ===
print("\n🌀 Starting Jad’s AI Intraday Bot (Every 5 min)\n")
while True:
    try:
        run_prediction()
    except Exception as e:
        print("❌ Error in cycle:", e)
    print(f"\n⏳ Waiting 5 minutes... ({time.ctime()})\n")
    time.sleep(300)
