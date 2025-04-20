import streamlit as st
import pandas as pd
import os
from datetime import datetime
from binance.client import Client
from binance_keys import API_KEY, API_SECRET

client = Client(API_KEY, API_SECRET)

st.set_page_config(page_title="Jad's AI Trading Dashboard", layout="wide")
st.title("📊 Jad’s AI Trading Dashboard")

# === LOAD TRADE LOG ===
def load_trades():
    try:
        df = pd.read_csv("executed_trades.csv", on_bad_lines='skip')
        if "Timestamp" not in df.columns or "Price" not in df.columns:
            st.warning("⚠️ Missing expected columns in executed_trades.csv.")
            return pd.DataFrame()
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors='coerce')
        df["Date"] = df["Timestamp"].dt.date
        df["Price"] = pd.to_numeric(df["Price"], errors='coerce')
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors='coerce')
        return df.dropna(subset=["Timestamp", "Price", "Quantity"])
    except Exception as e:
        st.error(f"Error loading trades: {e}")
        return pd.DataFrame()

# === PNL SUMMARY ===
def calculate_pnl(df):
    pnl_summary = []
    daily = {}
    buys = {}

    for _, row in df.iterrows():
        date = row["Date"]
        symbol = row["Symbol"]
        if date not in daily:
            daily[date] = 0

        if row["Side"] == "BUY":
            buys[symbol] = (row["Price"], row["Quantity"])
        elif row["Side"] == "SELL" and symbol in buys:
            buy_price, qty = buys[symbol]
            pnl = (row["Price"] - buy_price) * qty
            daily[date] += pnl
            del buys[symbol]

    for date, pnl in daily.items():
        pnl_summary.append({"Date": date, "Realized PnL (USD)": pnl})

    return pd.DataFrame(pnl_summary)

# === ASSET ALLOCATION ===
def get_asset_allocation():
    try:
        balances = client.get_account()["balances"]
        allocation = {
            b["asset"]: float(b["free"]) + float(b["locked"])
            for b in balances
            if float(b["free"]) + float(b["locked"]) > 0 and not b["asset"].startswith("LD")
        }
        return pd.DataFrame(list(allocation.items()), columns=["Asset", "Amount"])
    except Exception as e:
        st.error(f"Failed to fetch account balances: {e}")
        return pd.DataFrame()

# === AUM ===
def get_total_aum():
    try:
        total = 0
        prices = {s['symbol']: float(s['price']) for s in client.get_all_tickers()}
        balances = client.get_account()["balances"]
        for b in balances:
            asset = b["asset"]
            free = float(b["free"])
            locked = float(b["locked"])
            amount = free + locked
            if asset == "USDT":
                total += amount
            elif asset + "USDT" in prices:
                total += amount * prices[asset + "USDT"]
        return round(total, 2)
    except:
        return "N/A"

# === REFRESH ===
if st.button("🔄 Refresh Dashboard"):
    st.rerun()

# === DISPLAY SECTIONS ===
df_trades = load_trades()
df_pnl = calculate_pnl(df_trades)
df_allocation = get_asset_allocation()
aum = get_total_aum()

st.subheader(f"💰 Total AUM: ${aum}")

col1, col2 = st.columns(2)
with col1:
    st.subheader("📅 Daily Realized PnL")
    st.dataframe(df_pnl.sort_values("Date", ascending=False), use_container_width=True)
with col2:
    st.subheader("📦 Asset Allocation")
    if not df_allocation.empty:
        st.dataframe(df_allocation, use_container_width=True)
        st.bar_chart(data=df_allocation.set_index("Asset"))
    else:
        st.write("No active assets found.")

st.subheader("📜 Executed Trades")
st.dataframe(df_trades.sort_values("Timestamp", ascending=False), use_container_width=True)
