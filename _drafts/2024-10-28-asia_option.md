---
title : barrier option
date : 2024-10-28 11:24:29 +0800
categories: 
    - financial
    - option
---

**asian option**（亚洲期权）  
   - 定价：基于路径依赖的期权，价格取决于标的资产在一段时间内的平均价格。常用蒙特卡洛模拟或 PDE 方法。

<code>需要验证算法正确性</code>

```py
import numpy as np

def asian_option_monte_carlo(S0, K, T, r, sigma, n_steps, n_simulations):
    dt = T / n_steps
    payoff_sum = 0

    for _ in range(n_simulations):
        prices = [S0]
        for _ in range(n_steps):
            z = np.random.standard_normal()
            S_t = prices[-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
            prices.append(S_t)
        average_price = np.mean(prices)
        payoff = max(average_price - K, 0)
        payoff_sum += payoff

    option_price = np.exp(-r * T) * (payoff_sum / n_simulations)
    return option_price
```

```py
S0 = 100      # 初始股票价格
K = 100       # 行权价
T = 1.0         # 到期时间（以年计）
r = 0.05      # 无风险利率
sigma = 0.2   # 波动率
n_steps = 252 # 时间步数（例如，一年中的交易日）
n_simulations = 10000 # 蒙特卡罗模拟次数

price = asian_option_monte_carlo(S0, K, T, r, sigma, n_steps, n_simulations)
print(f"Asia Option : {price:.2f}")
```