import yfinance as yf
import pandas as pd

tickers = pd.read_csv('/home/geo/Downloads/geo/_posts/fin/futu/profitability_stock_selection.csv')

# 创建空列表存储数据
stock_data = []

for ticker in tickers:
    stock = yf.Ticker(ticker)
    
    # 获取股价（使用最近收盘价）
    price = stock.history(period='1d')['Close'].iloc[-1]
    
    # 获取每股股利（最近一年股息总额）
    dividend = stock.info.get('dividendRate', 0)  # 如果没有股息则为 0
    
    # 计算股息收益率
    dividend_yield = (dividend / price) * 100 if price > 0 else 0
    
    stock_data.append({
        'Ticker': ticker,
        'DPS': round(dividend_yield, 2)
    })

df = pd.DataFrame(stock_data)

df_sorted = df.sort_values(by='DPS', ascending=False)

print(df_sorted)

#%%

# https://chatgpt.com/c/694968c3-a168-8323-a6f2-d0d1c492791d