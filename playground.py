import yfinance as yf
import pandas as pd


msft = yf.Ticker("SOL1-USD")
data = msft.history(period="max")

start_date = data.index.values[0]
last_date = data.index.values[-1]

print(data.head())