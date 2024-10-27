---
title : barrier option 
date : 2024-10-27 11:24:29 +0800
categories: 
    - financial
    - option
---

### **barrier option**（障碍期权）  
   - 定价：当标的资产达到预定价格时，期权可能生效或失效。定价使用布莱克-舒尔斯模型结合反映障碍的条件。

存在解析解和 `monte carlo`

### 解析解

```py
import numpy as np
from scipy.stats import norm

def barrier_option_price(S, K, T, r, sigma, B, option_type, barrier_type):
    """
    计算四种障碍期权的价格

    参数:
    S: 初始资产价格
    K: 执行价格
    T: 到期时间
    r: 无风险利率
    sigma: 波动率
    B: 障碍价格
    option_type: 'call' 或 'put'，表示看涨或看跌
    barrier_type: 'up_and_out', 'down_and_out', 'up_and_in', 'down_and_in' 障碍类型

    返回:
    障碍期权价格
    """
    lambda_ = (r + 0.5 * sigma**2) / sigma**2
    x1 = np.log(S / K) / (sigma * np.sqrt(T)) + (1 + lambda_) * sigma * np.sqrt(T)
    x2 = np.log(S / B) / (sigma * np.sqrt(T)) + (1 + lambda_) * sigma * np.sqrt(T)
    y1 = np.log(B**2 / (S * K)) / (sigma * np.sqrt(T)) + (1 + lambda_) * sigma * np.sqrt(T)
    y2 = np.log(B / S) / (sigma * np.sqrt(T)) + (1 + lambda_) * sigma * np.sqrt(T)
    
    # 计算价格公式
    if barrier_type == 'up_and_out' and option_type == 'call':
        if S >= B:
            return 0.0
        return S * norm.cdf(x1) - K * np.exp(-r * T) * norm.cdf(x1 - sigma * np.sqrt(T)) - \
               S * (B / S)**(2 * lambda_) * norm.cdf(y1) + \
               K * np.exp(-r * T) * (B / S)**(2 * lambda_ - 2) * norm.cdf(y1 - sigma * np.sqrt(T))
    
    elif barrier_type == 'down_and_out' and option_type == 'put':
        if S <= B:
            return 0.0
        return K * np.exp(-r * T) * norm.cdf(-x1 + sigma * np.sqrt(T)) - S * norm.cdf(-x1) - \
               K * np.exp(-r * T) * (B / S)**(2 * lambda_ - 2) * norm.cdf(-y1 + sigma * np.sqrt(T)) + \
               S * (B / S)**(2 * lambda_) * norm.cdf(-y1)
    
    elif barrier_type == 'up_and_in' and option_type == 'call':
        if S >= B:
            # 价格即为普通看涨期权
            return S * norm.cdf(x1 - sigma * np.sqrt(T)) - K * np.exp(-r * T) * norm.cdf(x1 - sigma * np.sqrt(T))
        return (S * (B / S)**(2 * lambda_) * norm.cdf(y1) - \
                K * np.exp(-r * T) * (B / S)**(2 * lambda_ - 2) * norm.cdf(y1 - sigma * np.sqrt(T)))

    elif barrier_type == 'down_and_in' and option_type == 'put':
        if S <= B:
            # 价格即为普通看跌期权
            return K * np.exp(-r * T) * norm.cdf(-x1 + sigma * np.sqrt(T)) - S * norm.cdf(-x1)
        return (K * np.exp(-r * T) * (B / S)**(2 * lambda_ - 2) * norm.cdf(-y1 + sigma * np.sqrt(T)) - \
                S * (B / S)**(2 * lambda_) * norm.cdf(-y1))
    
    else:
        raise ValueError("不支持的期权类型或障碍类型")

# 示例参数
S = 100     # 初始资产价格
K = 100     # 执行价格
T = 1       # 到期时间（以年计）
r = 0.05    # 无风险利率
sigma = 0.2 # 波动率
B = 110     # 障碍价格

# 计算不同类型的障碍期权价格
price_up_and_out_call = barrier_option_price(S, K, T, r, sigma, B, option_type='call', barrier_type='up_and_out')
print("Up-and-Out Call Option Price:", price_up_and_out_call)

price_down_and_out_put = barrier_option_price(S, K, T, r, sigma, B, option_type='put', barrier_type='down_and_out')
print("Down-and-Out Put Option Price:", price_down_and_out_put)

price_up_and_in_call = barrier_option_price(S, K, T, r, sigma, B, option_type='call', barrier_type='up_and_in')
print("Up-and-In Call Option Price:", price_up_and_in_call)

price_down_and_in_put = barrier_option_price(S, K, T, r, sigma, B, option_type='put', barrier_type='down_and_in')
print("Down-and-In Put Option Price:", price_down_and_in_put)
```

### monte carlo

```py
import numpy as np

def monte_carlo_barrier_option(S, K, T, r, sigma, B, option_type='call', barrier_type='up_and_out', num_paths=10000, num_steps=100):
    """
    使用蒙特卡洛方法计算障碍期权的价格

    参数:
    S: 初始资产价格
    K: 执行价格
    T: 到期时间
    r: 无风险利率
    sigma: 波动率
    B: 障碍价格
    option_type: 'call' 或 'put'，代表看涨或看跌
    barrier_type: 'up_and_out' 或 'down_and_out'，代表障碍类型
    num_paths: 蒙特卡洛模拟路径数量
    num_steps: 每条路径的时间步数

    返回:
    障碍期权价格
    """
    dt = T / num_steps  # 每一步的时间间隔
    discount_factor = np.exp(-r * T)  # 折现因子
    payoff_sum = 0.0  # 总的期权收益

    for _ in range(num_paths):
        # 初始化路径
        path = [S]
        barrier_breached = False

        # 生成路径
        for _ in range(num_steps):
            z = np.random.standard_normal()
            S_t = path[-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
            path.append(S_t)

            # 检查是否触发障碍
            if barrier_type == 'up_and_out' and S_t >= B:
                barrier_breached = True
                break
            elif barrier_type == 'down_and_out' and S_t <= B:
                barrier_breached = True
                break

        # 如果障碍未触发，计算期权收益
        if not barrier_breached:
            if option_type == 'call':
                payoff = max(path[-1] - K, 0)  # 看涨期权
            elif option_type == 'put':
                payoff = max(K - path[-1], 0)  # 看跌期权
            payoff_sum += payoff

    # 计算期权价格（期望收益的折现值）
    price = (payoff_sum / num_paths) * discount_factor
    return price

# 示例参数
S = 100     # 初始资产价格
K = 100     # 执行价格
T = 1       # 到期时间（以年计）
r = 0.05    # 无风险利率
sigma = 0.2 # 波动率
B = 110     # 障碍价格

# 计算up-and-out call
price_up_and_out_call = monte_carlo_barrier_option(S, K, T, r, sigma, B, option_type='call', barrier_type='up_and_out')
print("Up-and-Out Call Option Price (Monte Carlo):", price_up_and_out_call)

# 计算down-and-out put
price_down_and_out_put = monte_carlo_barrier_option(S, K, T, r, sigma, B, option_type='put', barrier_type='down_and_out')
print("Down-and-Out Put Option Price (Monte Carlo):", price_down_and_out_put)
```