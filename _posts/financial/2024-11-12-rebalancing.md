---
layout: post
title : 再平衡策略 rebalancing
date : 2024-11-12 11:24:29 +0800
categories: 
    - financial
    - stock
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