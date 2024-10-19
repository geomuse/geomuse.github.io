---
layout: post
title:  fe black scholes model
date:   2024-10-09 11:24:29 +0800
categories: 
    - financial 
    - python
    - options
---

<!-- 在页面中直接加载 MathJax -->
<script type="text/javascript" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

### Black-Scholes

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

### **其他定价方法**

除了Black-Scholes公式，还有其他用于期权定价的方法：

1. **二叉树模型（Binomial Tree Model）**

   该模型将资产价格的可能路径离散化，构建一个价格树。在每个时间步，资产价格可以上升或下降。通过从到期日回溯计算，可以得到期权的当前价格。

   ```py
    def binomial_tree_option(S, K, T, r, sigma, N, option_type='call'):
        dt = T / N
        u = np.exp(sigma * np.sqrt(dt))      # 上升因子
        d = 1 / u                            # 下降因子
        p = (np.exp(r * dt) - d) / (u - d)   # 风险中性概率

        # 资产价格树
        asset_prices = np.zeros(N+1)
        for i in range(N+1):
            asset_prices[i] = S * (u ** (N - i)) * (d ** i)

        # 期权价值树
        option_values = np.zeros(N+1)
        if option_type == 'call':
            option_values = np.maximum(asset_prices - K, 0)
        else:
            option_values = np.maximum(K - asset_prices, 0)

        # 反向迭代计算期权价格
        for j in range(N-1, -1, -1):
            for i in range(j+1):
                option_values[i] = np.exp(-r * dt) * (p * option_values[i] + (1 - p) * option_values[i+1])

        return option_values[0]

    # 参数示例
    S = 100     # 资产现价
    K = 100     # 执行价格
    T = 1       # 到期时间（年）
    r = 0.05    # 无风险利率
    sigma = 0.2 # 波动率
    N = 50      # 时间步数

    price = binomial_tree_option(S, K, T, r, sigma, N, option_type='call')
    print(f"binomail tree : {price:.2f}")
   ```

2. **蒙特卡罗模拟（Monte Carlo Simulation）**

   通过模拟大量可能的资产价格路径，计算期权在每条路径上的收益，然后取平均值并折现到现值。

   ```py
    def monte_carlo_option(S, K, T, r, sigma, simulations, option_type='call'):
        dt = T
        discount_factor = np.exp(-r * T)
        payoff = np.zeros(simulations)

        for i in range(simulations):
            # 模拟资产价格路径
            Z = np.random.standard_normal()
            S_T = S * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

            # 计算期权收益
            if option_type == 'call':
                payoff[i] = max(S_T - K, 0)
            else:
                payoff[i] = max(K - S_T, 0)

        price = discount_factor * np.mean(payoff)
        return price

    # 参数示例
    simulations = 100000

    price = monte_carlo_option(S, K, T, r, sigma, simulations, option_type='call')
    print(f"monte carlo : {price:.2f}")

   ```

3. **有限差分方法（Finite Difference Methods）**

   该方法通过数值求解Black-Scholes偏微分方程，适用于定价复杂的衍生品。

   ```py
    def finite_difference_option(S, K, T, r, sigma, S_max, M, N, option_type='call'):
        dt = T / N
        dS = S_max / M
        i_values = np.arange(M+1)
        j_values = np.arange(N+1)
        grid = np.zeros((M+1, N+1))
        S_values = i_values * dS

        # 终端条件
        if option_type == 'call':
            grid[:, -1] = np.maximum(S_values - K, 0)
        else:
            grid[:, -1] = np.maximum(K - S_values, 0)

        # 边界条件
        if option_type == 'call':
            grid[-1, :] = S_max - K * np.exp(-r * dt * (N - j_values))
            grid[0, :] = 0
        else:
            grid[0, :] = K * np.exp(-r * dt * (N - j_values))
            grid[-1, :] = 0

        # 反向迭代
        alpha = 0.5 * dt * (sigma ** 2 * (i_values ** 2) - r * i_values)
        beta = 1 - dt * (sigma ** 2 * (i_values ** 2) + r)
        gamma = 0.5 * dt * (sigma ** 2 * (i_values ** 2) + r * i_values)

        for j in reversed(range(N)):
            for i in range(1, M):
                grid[i, j] = alpha[i] * grid[i-1, j+1] + beta[i] * grid[i, j+1] + gamma[i] * grid[i+1, j+1]

        # 插值找到期权价格
        price = np.interp(S, S_values, grid[:, 0])
        return price

    # 参数示例
    S_max = 200
    M = 100
    N = 100

    price = finite_difference_option(S, K, T, r, sigma, S_max, M, N, option_type='call')
    print(f"有限差分法 : {price:.2f}")
   ```

4. **特征函数方法（Characteristic Function Methods）**

   使用傅立叶变换或快速傅立叶变换（FFT）来计算期权价格，适用于处理复杂的资产价格分布。

   ```py
    def fft_option_price(S, K, T, r, sigma, alpha=1.5, N=4096, B=1000):
        eta = B / N
        lambd = (2 * np.pi) / (N * eta)
        beta = np.log(K)
        u = np.arange(N) * eta
        v = u - (alpha + 1) * 1j

        # 定义特征函数
        def char_func(v):
            return np.exp((1j * v * (np.log(S) + (r - 0.5 * sigma ** 2) * T)) - 0.5 * sigma ** 2 * v ** 2 * T)

        # 调整后的特征函数
        phi = np.exp(-r * T) * char_func(v) / (alpha ** 2 + alpha - v ** 2 + 1j * (2 * alpha + 1) * v)

        # 应用FFT
        payoff = np.exp(-1j * beta * u) * phi * eta
        payoff[0] = payoff[0] * 0.5  # 修正第一个元素
        fft_values = np.fft.fft(payoff).real
        strikes = np.exp(beta + lambd * np.arange(N))
        prices = fft_values / np.pi

        # 插值找到期权价格
        price = np.interp(K, strikes, prices)
        return price

    # 参数示例
    price = fft_option_price(S, K, T, r, sigma)
    print(f"fft option : {price:.2f}")
    ```   

5. **跳跃扩散模型（Jump Diffusion Models）**

   考虑资产价格的跳跃行为，更贴近真实市场情况，如Merton模型。

   ```py
    def merton_jump_diffusion_option(S, K, T, r, sigma, lam, m, v, simulations, option_type='call'):
        dt = T
        discount_factor = np.exp(-r * T)
        payoff = np.zeros(simulations)

        for i in range(simulations):
            # 模拟跳跃次数
            N_J = np.random.poisson(lam * T)
            # 模拟跳跃大小
            Y = np.random.normal(m, v, N_J)
            # 计算跳跃乘积
            J = np.prod(np.exp(Y)) if N_J > 0 else 1.0
            # 模拟资产价格路径
            Z = np.random.standard_normal()
            S_T = S * J * np.exp((r - 0.5 * sigma ** 2 - lam * (np.exp(m + 0.5 * v ** 2) - 1)) * dt + sigma * np.sqrt(dt) * Z)

            # 计算期权收益
            if option_type == 'call':
                payoff[i] = max(S_T - K, 0)
            else:
                payoff[i] = max(K - S_T, 0)

        price = discount_factor * np.mean(payoff)
        return price

    # 参数示例
    lam = 0.1     # 跳跃强度
    m = -0.1      # 跳跃幅度的均值
    v = 0.2       # 跳跃幅度的标准差
    simulations = 100000

    price = merton_jump_diffusion_option(S, K, T, r, sigma, lam, m, v, simulations, option_type='call')
    print(f"jump diffusion : {price:.2f}")
   ```

6. **局部波动率模型（Local Volatility Models）**

   假设波动率是资产价格和时间的函数，如Dupire模型。

7. **随机波动率模型（Stochastic Volatility Models）**

   假设波动率本身也是随机过程，如Heston模型。

   ```py
    def heston_model_option(S, K, T, r, v0, kappa, theta, sigma, rho, simulations, steps, option_type='call'):
        dt = T / steps
        discount_factor = np.exp(-r * T)
        payoff = np.zeros(simulations)

        for i in range(simulations):
            S_t = S
            v_t = v0
            for _ in range(steps):
                Z1 = np.random.standard_normal()
                Z2 = np.random.standard_normal()
                Z2 = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2

                # 更新波动率
                v_t = np.abs(v_t + kappa * (theta - v_t) * dt + sigma * np.sqrt(v_t * dt) * Z2)
                # 更新资产价格
                S_t = S_t * np.exp((r - 0.5 * v_t) * dt + np.sqrt(v_t * dt) * Z1)

            # 计算期权收益
            if option_type == 'call':
                payoff[i] = max(S_t - K, 0)
            else:
                payoff[i] = max(K - S_t, 0)

        price = discount_factor * np.mean(payoff)
        return price

    # 参数示例
    v0 = 0.04     # 初始方差
    kappa = 2.0   # 均值回复速度
    theta = 0.04  # 长期方差均值
    sigma = 0.2   # 波动率的波动率
    rho = -0.7    # 资产价格和波动率之间的相关系数
    simulations = 10000
    steps = 100

    price = heston_model_option(S, K, T, r, v0, kappa, theta, sigma, rho, simulations, steps, option_type='call')
    print(f"heston models : {price:.2f}")
   ```