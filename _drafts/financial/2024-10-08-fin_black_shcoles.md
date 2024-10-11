---
layout: post
title:  fe black scholes model
date:   2024-10-08 11:24:29 +0800
categories: 
    - financial 
    - python
---

<!-- 在页面中直接加载 MathJax -->
<script type="text/javascript" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

Black-Scholes公式用于计算欧式看涨和看跌期权的理论价格。公式如下：

对于看涨期权（Call Option）：

$$ C = S_0 N(d_1) - Ke^{-rT} N(d_2) $$

对于看跌期权（Put Option）：

$$ P = Ke^{-rT} N(-d_2) - S_0 N(-d_1) $$

其中：

$$ d_1 = \frac{\ln(S_0 / K) + (r + \frac{\sigma^2}{2})T}{\sigma \sqrt{T}} $$

$$d_2 = d_1 - \sigma \sqrt{T} $$

$$ S_o $$ : 当前资产价格

$$K$$：行权价

$$ r $$：无风险利率

$$ T $$：到期时间（以年为单位）

$$ \sigma $$：资产的波动率

$$ N(\cdot) $$：标准正态分布的累积分布函数

**Python实现**

以下是使用Python实现Black-Scholes公式的示例代码：

```python
import numpy as np
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    计算欧式期权价格。

    参数：
    S : float
        标的资产当前价格
    K : float
        行权价
    T : float
        到期时间（以年为单位）
    r : float
        无风险利率
    sigma : float
        波动率
    option_type : str
        'call'表示看涨期权，'put'表示看跌期权

    返回：
    price : float
        期权价格
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type参数必须是'call'或'put'。")

    return price

# 示例用法
S0 = 100    # 当前资产价格
K = 100     # 行权价
T = 1       # 到期时间1年
r = 0.05    # 无风险利率5%
sigma = 0.2 # 波动率20%

call_price = black_scholes(S0, K, T, r, sigma, option_type='call')
put_price = black_scholes(S0, K, T, r, sigma, option_type='put')

print(f"看涨期权价格：{call_price:.2f}")
print(f"看跌期权价格：{put_price:.2f}")
```

**其他定价方法**

除了Black-Scholes公式，还有其他用于期权定价的方法：

1. **二叉树模型（Binomial Tree Model）**

   该模型将资产价格的可能路径离散化，构建一个价格树。在每个时间步，资产价格可以上升或下降。通过从到期日回溯计算，可以得到期权的当前价格。

2. **蒙特卡罗模拟（Monte Carlo Simulation）**

   通过模拟大量可能的资产价格路径，计算期权在每条路径上的收益，然后取平均值并折现到现值。

3. **有限差分方法（Finite Difference Methods）**

   该方法通过数值求解Black-Scholes偏微分方程，适用于定价复杂的衍生品。

4. **特征函数方法（Characteristic Function Methods）**

   使用傅立叶变换或快速傅立叶变换（FFT）来计算期权价格，适用于处理复杂的资产价格分布。

5. **跳跃扩散模型（Jump Diffusion Models）**

   考虑资产价格的跳跃行为，更贴近真实市场情况，如Merton模型。

6. **局部波动率模型（Local Volatility Models）**

   假设波动率是资产价格和时间的函数，如Dupire模型。

7. **随机波动率模型（Stochastic Volatility Models）**

   假设波动率本身也是随机过程，如Heston模型。