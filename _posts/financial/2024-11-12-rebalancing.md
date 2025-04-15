---
layout: post
title : rebalancing 再平衡策略 
date : 2024-11-12 11:24:29 +0800
categories: 
    - financial
    - portfolio
---

```
初始持仓: {'Stock_A': 40.0, 'Stock_B': 60.0, 'Stock_C': 15.0}
第1天组合总市值：10000.00
第2天组合总市值：10170.00
第2天再平衡后的持仓: {'Stock_A': 39.88235294117647, 'Stock_B': 59.8235294117647, 'Stock_C': 15.103960396039604}
第3天组合总市值：10145.52
第4天组合总市值：10390.54
第4天再平衡后的持仓: {'Stock_A': 40.35160219619906, 'Stock_B': 59.9454090318534, 'Stock_C': 14.843625093601796}
第5天组合总市值：10560.87
再平衡策略完成后的组合价值：
            Portfolio Value
2023-01-01     10000.000000
2023-01-02     10170.000000
2023-01-03     10145.517764
2023-01-04     10390.537566
2023-01-05     10560.873429
```

```py
import pandas as pd
import numpy as np

data = pd.DataFrame({
    'Stock_A': [100, 102, 101, 103, 105],
    'Stock_B': [50, 51, 50.5, 52, 53],
    'Stock_C': [200, 202, 205, 210, 212],
}, index=pd.date_range(start='2023-01-01', periods=5))

initial_investment = 10000
initial_weights = {
    'Stock_A': 0.4,
    'Stock_B': 0.3,
    'Stock_C': 0.3
}

# 计算初始持仓
holdings = {stock: initial_investment * weight / data[stock].iloc[0] for stock, weight in initial_weights.items()}
print("初始持仓:", holdings)

rebalance_period = 2

portfolio_values = []

for i in range(len(data)):
    # 当前组合市值
    current_values = {stock: holdings[stock] * data[stock].iloc[i] for stock in holdings}
    total_portfolio_value = sum(current_values.values())
    portfolio_values.append(total_portfolio_value)

    print(f"第{i+1}天组合总市值：{total_portfolio_value:.2f}")
    
    if (i + 1) % rebalance_period == 0:
        for stock, weight in initial_weights.items():
            holdings[stock] = (total_portfolio_value * weight) / data[stock].iloc[i]
        print(f"第{i+1}天再平衡后的持仓:", holdings)

result = pd.DataFrame({'Portfolio Value': portfolio_values}, index=data.index)

print("再平衡策略完成后的组合价值：")
print(result)
```

```py
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as pt

# 1. 下载资产数据
tickers = ['AAPL', 'MSFT', 'GOOGL']  # 示例股票
data = yf.download(tickers, start='2020-01-01', end='2024-01-01')['Adj Close']

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
```