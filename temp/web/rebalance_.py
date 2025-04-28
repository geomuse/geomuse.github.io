import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# 设置股票和回测时间
tickers = ['AAPL', 'MSFT', 'GOOGL']
start_date = '2020-01-01'
end_date = '2024-12-31'

# 下载数据
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# 每月最后一个交易日
monthly = data.resample('M').last()

# 初始化权重（等权）
n_assets = len(tickers)
weights = [1/n_assets] * n_assets

# 初始化组合价值
initial_value = 100000
portfolio_values = []

# 开始再平衡模拟
current_shares = None

for date in monthly.index:
    prices = monthly.loc[date]
    
    if current_shares is None:
        # 第一次买入
        current_shares = (initial_value * pd.Series(weights, index=tickers)) / prices
    else:
        # 再平衡：根据当前价格重新分配资金
        current_value = (current_shares * prices).sum()
        current_shares = (current_value * pd.Series(weights, index=tickers)) / prices

    # 记录组合价值
    portfolio_value = (current_shares * prices).sum()
    portfolio_values.append(portfolio_value)

# 绘图
portfolio_series = pd.Series(portfolio_values, index=monthly.index)
portfolio_series.plot(figsize=(10, 5), title='等权再平衡组合表现')
plt.ylabel('组合价值 ($)')
plt.grid(True)
plt.show()
