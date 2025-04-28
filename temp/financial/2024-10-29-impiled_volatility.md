---
layout: post
title:  impiled volatility
date:   2024-10-29 11:24:29 +0800
categories: 
    - financial
    - volatility
---

<script>
  MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']]
    }
  };
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

要使用二分法（Bisection）、割线法（Secant），和牛顿法（Newton）来求取期权的隐含波动率（Implied Volatility），我们可以借助 Python 代码实现。这里会用到以下步骤：

1. 设置目标函数：根据期权定价公式（如 Black-Scholes），计算理论价格，并将理论价格与市场实际价格的差异作为目标函数。

2. 实现方法：分别使用二分法、割线法、牛顿法求取使目标函数等于零的隐含波动率。

1. 设置 Black-Scholes 公式的目标函数

首先，计算期权的理论价格。我们使用 Black-Scholes 模型来求得目标价格，并设定市场价格与理论价格之差为目标函数。

```py
import numpy as np
from scipy.stats import norm

def black_scholes_call_price(S, K, T, r, sigma):
    """计算欧式看涨期权的 Black-Scholes 价格"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def objective_function(sigma, S, K, T, r, market_price):
    """目标函数：Black-Scholes价格与市场价格的差值"""
    bs_price = black_scholes_call_price(S, K, T, r, sigma)
    return bs_price - market_price
```

2. 二分法求解隐含波动率

二分法需要初始的上下限，我们可以用 0.0001 和 5 作为波动率的范围。
```py
def implied_volatility_bisection(S, K, T, r, market_price, tol=1e-6, max_iterations=100):
    lower, upper = 0.0001, 5
    for i in range(max_iterations):
        mid = (lower + upper) / 2
        price_diff = objective_function(mid, S, K, T, r, market_price)
        
        if abs(price_diff) < tol:
            return mid
        elif price_diff > 0:
            upper = mid
        else:
            lower = mid
    return (lower + upper) / 2  # 超过迭代次数后返回中间值
```

3. 割线法求解隐含波动率

割线法需要两个初始猜测，我们可以用 0.0001 和 0.5 作为初始猜测值。

```py
def implied_volatility_secant(S, K, T, r, market_price, tol=1e-6, max_iterations=100):
    x0, x1 = 0.0001, 0.5
    for i in range(max_iterations):
        f_x0 = objective_function(x0, S, K, T, r, market_price)
        f_x1 = objective_function(x1, S, K, T, r, market_price)
        
        if abs(f_x1) < tol:
            return x1
        
        # 割线更新公式
        x_new = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        
        x0, x1 = x1, x_new
    return x1  # 超过迭代次数后返回最后一次计算值
```
4. 牛顿法求解隐含波动率

牛顿法要求计算 Black-Scholes 价格对波动率的导数。

```py
def vega(S, K, T, r, sigma):
    """Black-Scholes 公式的 Vega，用于牛顿法中的导数"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def implied_volatility_newton(S, K, T, r, market_price, tol=1e-6, max_iterations=100):
    sigma = 0.5  # 初始猜测值
    for i in range(max_iterations):
        price_diff = objective_function(sigma, S, K, T, r, market_price)
        vega_val = vega(S, K, T, r, sigma)
        
        if abs(price_diff) < tol:
            return sigma
        
        sigma -= price_diff / vega_val  # 牛顿法更新公式
    return sigma  # 超过迭代次数后返回最后一次计算值
```

示例代码

使用上述函数来求解隐含波动率。

```py
# 输入期权参数
S = 100          # 标的资产价格
K = 100          # 行权价
T = 1            # 距到期时间（年）
r = 0.05         # 无风险利率
market_price = 10  # 市场价格

# 计算隐含波动率
iv_bisection = implied_volatility_bisection(S, K, T, r, market_price)
iv_secant = implied_volatility_secant(S, K, T, r, market_price)
iv_newton = implied_volatility_newton(S, K, T, r, market_price)

print("隐含波动率 (Bisection):", iv_bisection)
print("隐含波动率 (Secant):", iv_secant)
print("隐含波动率 (Newton):", iv_newton)
```