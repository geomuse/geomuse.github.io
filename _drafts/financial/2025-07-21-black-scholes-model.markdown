---
layout: post
title:  black scholes model
date:   2025-07-21 11:01:35 +0300
image:  03.jpg
tags:   Financial Option Python
---

### Black-Scholes 模型（Black-Scholes-Merton Model）

#### 假设条件：

* 标的资产价格服从几何布朗运动（GBM）
* 无风险利率是恒定的
* 波动率是恒定的
* 无交易成本或税收
* 可做空
* 欧式期权（只能在到期时执行）

衍生品定价与对冲问题 是金融工程中的核心问题，尤其是在不完整市场条件下，即存在交易成本、跳跃扩散过程和随机波动性等因素时，传统的Black-Scholes-Merton模型假设已经不再成立。
传统的动态规划和效用优化方法难以有效处理复杂市场摩擦

#### Black-Scholes 欧式看涨期权定价公式：

$$
C = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2)
$$

$$
d_1 = \frac{\ln(S_0 / K) + (r - q + \frac{1}{2} \sigma^2)T}{\sigma \sqrt{T}}, \quad
d_2 = d_1 - \sigma \sqrt{T}
$$

其中：

* $C$：看涨期权价格
* $S_0$：标的资产当前价格
* $K$：执行价格
* $T$：到期时间
* $r$：无风险利率
* $q$：股息收益率
* $\sigma$：波动率
* $N(\cdot)$：标准正态分布累积分布函数

#### Black-Scholes 欧式看涨期权定价公式 Python

```py
import numpy as np
import scipy.stats as stats 

def black_scholes():
    d1 = (np.log(so / k) + (r + 0.5 * volatility ** 2) * T) / (volatility * np.sqrt(T))
    d2 = (np.log(so / k) + (r - 0.5 * volatility ** 2) * T) / (volatility * np.sqrt(T))
    value = (so * stats.norm.cdf(d1, 0.0, 1.0)- k * np.exp(-r * T) * stats.norm . cdf(d2, 0.0, 1.0))
    return value
```