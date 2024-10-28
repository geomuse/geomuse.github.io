---
title : asia option
date : 2024-10-28 11:24:29 +0800
categories: 
    - financial
    - option
---

**asian option**（亚洲期权）  
   - 定价：基于路径依赖的期权，价格取决于标的资产在一段时间内的平均价格。常用蒙特卡洛模拟或 PDE 方法。

<code>需要验证算法正确性与其他方法</code>

```py
from scipy.stats import norm
import numpy as np

def black_scholes_average_price_asian(S0, K, T, r, sigma, q=0, option_type='call'):
    """
    S0: 标的资产的初始价格
    K: 行权价格
    T: 到期时间（以年计）
    r: 无风险利率
    sigma: 波动率
    q: 连续股息收益率
    option_type: 'call' or 'put'
    """
    # 修正后的波动率
    sigma_avg = sigma * np.sqrt((2 * T + 1) / (6 * T))
    
    # 计算 d1 和 d2
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma_avg ** 2) * T) / (sigma_avg * np.sqrt(T))
    d2 = d1 - sigma_avg * np.sqrt(T)
    
    # 计算期权价格
    if option_type == 'call':
        option_price = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)
    
    return option_price

# 参数示例
S0 = 100  # 初始价格
K = 105   # 行权价格
T = 1.0   # 到期时间（1年）
r = 0.05  # 无风险利率
sigma = 0.2  # 波动率
q = 0.0  # 连续股息收益率
option_type = 'call'

price = black_scholes_average_price_asian(S0, K, T, r, sigma, q, option_type)
print("Average Price Asian Option Price:", price)
```

```py
def black_scholes_average_strike_asian(S0, S_avg, T, r, sigma, q=0, option_type='call'):
    """
    S0: 标的资产的初始价格
    S_avg: 行权价格的平均值
    T: 到期时间（以年计）
    r: 无风险利率
    sigma: 波动率
    q: 连续股息收益率
    option_type: 'call' or 'put'
    """
    # 修正后的波动率
    sigma_avg = sigma * np.sqrt((2 * T + 1) / (6 * T))
    
    # 计算 d1 和 d2
    d1 = (np.log(S0 / S_avg) + (r - q + 0.5 * sigma_avg ** 2) * T) / (sigma_avg * np.sqrt(T))
    d2 = d1 - sigma_avg * np.sqrt(T)
    
    # 计算期权价格
    if option_type == 'call':
        option_price = S0 * np.exp(-q * T) * norm.cdf(d1) - S_avg * np.exp(-r * T) * norm.cdf(d2)
    else:
        option_price = S_avg * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)
    
    return option_price

# 参数示例
S0 = 100      # 初始价格
S_avg = 105   # 平均行权价格
T = 1.0       # 到期时间（1年）
r = 0.05      # 无风险利率
sigma = 0.2   # 波动率
q = 0.0       # 连续股息收益率
option_type = 'call'

price = black_scholes_average_strike_asian(S0, S_avg, T, r, sigma, q, option_type)
print("Average Strike Asian Option Price:", price)
```

```py
import numpy as np

def monte_carlo_asian_option(S0, K, T, r, sigma, M, N, option_type='call'):
    """
    S0: 标的资产的初始价格
    K: 行权价格
    T: 到期时间（以年计）
    r: 无风险利率
    sigma: 波动率
    M: 时间步数
    N: 模拟次数
    option_type: 'call' or 'put'
    """
    dt = T / M
    payoff = []
    
    for _ in range(N):
        # 模拟价格路径
        S = [S0]
        for _ in range(1, M + 1):
            S_t = S[-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal())
            S.append(S_t)
        
        # 计算平均价格
        S_avg = np.mean(S)
        
        # 计算期权的收益
        if option_type == 'call':
            payoff.append(max(S_avg - K, 0))
        else:
            payoff.append(max(K - S_avg, 0))
    
    # 计算期权价格
    option_price = np.exp(-r * T) * np.mean(payoff)
    return option_price

# 参数示例
S0 = 100  # 初始价格
K = 105   # 行权价格
T = 1.0   # 到期时间（1年）
r = 0.05  # 无风险利率
sigma = 0.2  # 波动率
M = 252   # 时间步数
N = 10000  # 模拟次数
option_type = 'call'

price = monte_carlo_asian_option(S0, K, T, r, sigma, M, N, option_type)
print("Average Price Asian Option Price:", price)
```

```py
class asian_option :

    def monte_carlo_call(self,so, K, T, r, sigma, n_steps, n_simulations):
        dt = T / n_steps
        payoff_sum = 0

        for _ in range(n_simulations):
            prices = [so]
            for _ in range(n_steps):
                z = np.random.standard_normal()
                S_t = prices[-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
                prices.append(S_t)
            average_price = np.mean(prices)
            payoff = max(average_price - K, 0)
            payoff_sum += payoff

        option_price = np.exp(-r * T) * (payoff_sum / n_simulations)
        return option_price

    def monte_carlo_put(so, K, T, r, sigma, n_steps, n_simulations):
        dt = T / n_steps
        payoff_sum = 0

        for _ in range(n_simulations):
            prices = [so]
            for _ in range(n_steps):
                z = np.random.standard_normal()
                S_t = prices[-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
                prices.append(S_t)
            average_price = np.mean(prices)
            payoff = max(K - average_price,0)
            payoff_sum += payoff

        option_price = np.exp(-r * T) * (payoff_sum / n_simulations)
        return option_price

S0 = 100      # 初始股票价格
K = 100       # 行权价
T = 1.0         # 到期时间（以年计）
r = 0.05      # 无风险利率
sigma = 0.2   # 波动率
n_steps = 252 # 时间步数（例如，一年中的交易日）
n_simulations = 10000 # 蒙特卡罗模拟次数

price = asian_option().monte_carlo_call(S0, K, T, r, sigma, n_steps, n_simulations)
print(f"Asia Option : {price:.2f}")
```