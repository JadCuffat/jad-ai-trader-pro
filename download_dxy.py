import yfinance as yf
import pandas as pd

def download_dxy(start="2022-01-01", end=None):
    dxy = yf.download("DX-Y.NYB", start=start, end=end, interval="1d")
    dxy.reset_index(inplace=True)
    dxy = dxy[["Date", "Open", "High", "Low", "Close", "Volume"]]
    dxy.to_csv("DXY_historical.csv", index=False)
    print("✅ DXY data saved to DXY_historical.csv")

download_dxy()
