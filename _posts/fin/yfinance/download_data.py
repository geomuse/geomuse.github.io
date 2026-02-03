# save as create_yahoo_bundle.py
import pandas as pd
import yfinance as yf
import os

tickers = ['AAPL', 'MSFT', 'GOOG',"GC=F","BTC-USD"]

data_dir = '/home/geo/Downloads/geo/_posts/fin/mt4/data/daily'
os.makedirs(data_dir, exist_ok=True)

for ticker in tickers:
    if ticker == 'BTC-USD' :
        df = yf.download("BTC-USD", period="7d", interval="1m")
        print(df)
        df.to_csv(os.path.join(data_dir, f'{ticker}.csv'))
    else :     
        df = yf.download(ticker, start='2020-01-01', end='2026-01-30',interval="1D")
        print(df)
        df.to_csv(os.path.join(data_dir, f'{ticker}.csv'))
