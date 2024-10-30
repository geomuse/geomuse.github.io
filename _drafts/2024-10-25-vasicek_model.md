---
layout: post
title : vasicek model
author : geo
date : 2024-10-25 11:24:29 +0800
categories: 
    - financial
    - bond
---

```py
import numpy as np
import matplotlib.pyplot as pt

# Vasicek模型参数
a = 0.1  # 均值回复速率
b = 0.05  # 长期均值
sigma = 0.02  # 利率波动率
r0 = 0.03  # 初始利率
T = 1.0  # 模拟期限（1年）
dt = 0.01  # 时间步长
n = int(T/dt)  # 时间步数

# 用于存储利率的数组
rates = np.zeros(n)
rates[0] = r0

# 布朗运动的随机变量
random_shocks = np.random.normal(0, 1, n)

# Vasicek模型的SDE模拟
for t in range(1, n):
    dr = a * (b - rates[t-1]) * dt + sigma * np.sqrt(dt) * random_shocks[t]
    rates[t] = rates[t-1] + dr

# 绘制模拟的短期利率曲线
pt.plot(np.linspace(0, T, n), rates)
pt.title("Vasicek Model: Short-term Interest Rate Simulation")
pt.xlabel("Time (Years)")
pt.ylabel("Interest Rate")
pt.grid(True)
pt.show()
```

可以通过Vasicek模型计算零息债券的价格

```py
def bond_price_vasicek(r, a, b, sigma, T):
    """
    计算零息债券价格，基于Vasicek模型
    :param r: 当前短期利率
    :param a: 均值回复速率
    :param b: 长期均值
    :param sigma: 波动率
    :param T: 到期时间
    :return: 债券价格
    """
    B = (1 - np.exp(-a * T)) / a
    A = np.exp((b - (sigma ** 2) / (2 * a ** 2)) * (B - T) - (sigma ** 2) * B ** 2 / (4 * a))
    P = A * np.exp(-r * B)
    return P

# 假设当前短期利率 r = 0.03，债券到期时间为 2 年
r_current = 0.03
T_bond = 2.0
bond_price = bond_price_vasicek(r_current, a, b, sigma, T_bond)
print(f"债券价格: {bond_price:.4f}")
```