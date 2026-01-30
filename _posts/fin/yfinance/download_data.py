# save as create_yahoo_bundle.py
import pandas as pd
import yfinance as yf
import os

tickers = ['AAPL', 'MSFT', 'GOOG']
data_dir = '/home/geo/Downloads/geo/_posts/fin/mt4/data/daily'
os.makedirs(data_dir, exist_ok=True)

for ticker in tickers:
    df = yf.download(ticker, start='2020-01-01', end='2026-01-30')
    df.to_csv(os.path.join(data_dir, f'{ticker}.csv'))
