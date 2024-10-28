---
title : binary option
date : 2024-10-28 11:24:29 +0800
categories: 
    - financial
    - option
---

**binary option**（二元期权）  
    - 定价：在到期时，价格要么是固定的金额，要么为零。可以使用布莱克-舒尔斯模型的修改版本定价。

在金融中，二元期权（Binary Option）是一种“全有或全无”的期权，其收益要么是固定的预定金额(若期权到期时处于盈余状态)，要么是零(若期权到期时处于亏损状态),常见的二元期权包括：

1. 现金或无（Cash-or-Nothing）看涨/看跌期权：若标的资产价格高于执行价格(对于看涨期权)或低于执行价格(对于看跌期权)，支付固定金额，否则支付为零。

2. 资产或无（Asset-or-Nothing）看涨/看跌期权：若标的资产价格高于执行价格(看涨期权)或低于执行价格(看跌期权)，支付标的资产的价格，否则支付为零。

### 解析解

```py
import numpy as np
from scipy.stats import norm

def binary_option_price(S, K, T, r, sigma, option_type='cash', payout=1, option_direction='call'):

    d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))  # 用于计算标准正态分布
    d1 = d2 + sigma * np.sqrt(T)  # d1 用于资产或无期权的计算

    if option_type == 'cash':
        # 现金或无二元期权
        if option_direction == 'call':
            price = payout * np.exp(-r * T) * norm.cdf(d2)  # 看涨期权支付
        elif option_direction == 'put':
            price = payout * np.exp(-r * T) * norm.cdf(-d2)  # 看跌期权支付

    elif option_type == 'asset':
        # 资产或无二元期权
        if option_direction == 'call':
            price = S * norm.cdf(d1)  # 看涨期权支付
        elif option_direction == 'put':
            price = S * norm.cdf(-d1)  # 看跌期权支付
    else:
        raise ValueError("option_type 必须为 'cash' 或 'asset'")
    
    return price

# 示例参数
S = 100     # 当前资产价格
K = 100     # 执行价格
T = 1       # 到期时间（以年计）
r = 0.05    # 无风险利率
sigma = 0.2 # 波动率
payout = 10 # 现金或无的固定支付金额

# 计算现金或无二元期权价格
print("Cash-or-Nothing Call Option Price:", binary_option_price(S, K, T, r, sigma, option_type='cash', payout=payout, option_direction='call'))
print("Cash-or-Nothing Put Option Price:", binary_option_price(S, K, T, r, sigma, option_type='cash', payout=payout, option_direction='put'))

# 计算资产或无二元期权价格
print("Asset-or-Nothing Call Option Price:", binary_option_price(S, K, T, r, sigma, option_type='asset', option_direction='call'))
print("Asset-or-Nothing Put Option Price:", binary_option_price(S, K, T, r, sigma, option_type='asset', option_direction='put'))
```

### monte carlo

```py
import numpy as np

def monte_carlo_binary_option(S0, K, T, r, sigma, num_simulations=10000):
    # 模拟末端价格
    Z = np.random.standard_normal(num_simulations)  # 随机变量
    ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    
    # 计算支付值
    payoff = np.where(ST > K, 1, 0)  # 若资产价格高于行权价，则支付1，否则支付0
    
    # 计算二元期权价格（现值）
    binary_option_price = np.exp(-r * T) * np.mean(payoff)
    
    return binary_option_price

# 参数设置
S0 = 100       # 初始资产价格
K = 105        # 行权价
T = 1          # 到期期限（1年）
r = 0.05       # 无风险利率
sigma = 0.2    # 波动率

# 计算二元期权价格
price = monte_carlo_binary_option(S0, K, T, r, sigma)
print(f"二元期权价格: {price:.4f}")
```