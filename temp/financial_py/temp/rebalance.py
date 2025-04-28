import pandas as pd
import numpy as np
from pymongo import MongoClient
import yfinance as yf
import matplotlib.pyplot as pt

# 1. 下载资产数据
tickers = ['AAPL', 'MSFT', 'GOOGL']  # 示例股票
data = yf.download(tickers, start='2020-01-01', end='2024-01-01')['Adj Close']

client = MongoClient("mongodb://localhost:27017/")
# 选择数据库（例如 "mydatabase"）
db = client["stock_data"]
collection_aapl = db["AAPL"]
collection_aapl = db["GOOGL"]
collection_aapl = db["MSFT"]

# 查询数据，例如：读取所有数据




# 2. 设置初始权重
target_weights = {'AAPL': 0.4, 'MSFT': 0.3, 'GOOGL': 0.3}

# 3. 计算每日收益率
returns = data.pct_change().dropna()

# 4. 初始化
initial_cash = 100000
portfolio_value = initial_cash
portfolio = {ticker: 0 for ticker in tickers}
values = []
dates = []
rebalancing_dates = returns.resample('M').first().index  # 每月第一个交易日

# 5. 回测循环
for date, row in returns.iterrows():
    # 如果是再平衡日
    if date in rebalancing_dates:
        portfolio_value = sum(data.loc[date][ticker] * portfolio[ticker] for ticker in tickers)
        for ticker in tickers:
            allocation = target_weights[ticker] * portfolio_value
            portfolio[ticker] = allocation // data.loc[date][ticker]

    # 更新投资组合价值
    portfolio_value = sum(data.loc[date][ticker] * portfolio[ticker] for ticker in tickers)
    values.append(portfolio_value)
    dates.append(date)

# 6. 绘图
pt.figure(figsize=(12, 6))
pt.plot(dates, values, label='Portfolio Value')
pt.title('Rebalancing Portfolio Strategy')
pt.xlabel('Date')
pt.ylabel('Portfolio Value ($)')
pt.legend()
pt.grid()
pt.show()
