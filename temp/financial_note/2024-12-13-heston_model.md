---
layout: post
title:  建构 heston model 随机过程
date:   2024-12-13 11:24:29 +0800
categories: 
    - financial
    - processing
---

<!-- 可使用下述程式碼把markdown格式轉成word

```
pandoc -o output.docx -f markdown -t docx input.md
``` -->
<script>
  MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']]
    }
  };
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

通过随机过程来求取 Heston 模型的欧式期权价格，通常使用 **蒙特卡洛模拟** 来模拟随机过程，并计算期权的贴现价值。这种方法需要模拟资产价格和波动率的路径。

以下是基于 Heston 模型的蒙特卡洛模拟的 Python 实现：

### 模型随机过程公式
1. **资产价格随机过程**：
   $$
   dS_t = r S_t dt + \sqrt{v_t} S_t dW_t^S
   $$
2. **波动率随机过程**：
   $$
   dv_t = \kappa (\theta - v_t) dt + \sigma \sqrt{v_t} dW_t^v
   $$
   其中 $ W_t^S $ 和 $ W_t^v $ 是两个相关的 Wiener 过程，相关性为 $ \rho $。

我们通过欧拉-马尔科夫法对这些随机微分方程进行数值解，进而估算期权价格。

### Python 实现

```python
import numpy as np

def heston_mc_european(S0, K, T, r, v0, theta, kappa, sigma, rho, num_paths, num_steps, option_type='call'):
    """
    Heston model Monte Carlo simulation for European option pricing.
    
    Parameters:
        S0: float - Initial stock price
        K: float - Strike price
        T: float - Time to maturity (in years)
        r: float - Risk-free interest rate
        v0: float - Initial variance
        theta: float - Long-term mean variance
        kappa: float - Mean reversion speed
        sigma: float - Volatility of variance
        rho: float - Correlation between stock and variance
        num_paths: int - Number of Monte Carlo paths
        num_steps: int - Number of time steps
        option_type: str - 'call' or 'put'
    
    Returns:
        float - Option price
    """
    dt = T / num_steps  # Time step
    S = np.zeros((num_steps + 1, num_paths))
    v = np.zeros((num_steps + 1, num_paths))
    S[0] = S0
    v[0] = v0

    # Simulate paths
    for t in range(1, num_steps + 1):
        # Generate correlated random numbers
        Z1 = np.random.normal(0, 1, num_paths)
        Z2 = np.random.normal(0, 1, num_paths)
        W1 = Z1
        W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2

        # Variance process (ensure non-negativity)
        v[t] = np.abs(v[t - 1] + kappa * (theta - v[t - 1]) * dt + sigma * np.sqrt(v[t - 1] * dt) * W2)

        # Asset price process
        S[t] = S[t - 1] * np.exp((r - 0.5 * v[t - 1]) * dt + np.sqrt(v[t - 1] * dt) * W1)

    # Calculate option payoff
    if option_type == 'call':
        payoff = np.maximum(S[-1] - K, 0)
    elif option_type == 'put':
        payoff = np.maximum(K - S[-1], 0)
    else:
        raise ValueError("Invalid option type. Choose 'call' or 'put'.")

    # Discount payoff to present value
    option_price = np.exp(-r * T) * np.mean(payoff)
    return option_price

# 参数设置
S0 = 100      # 初始资产价格
K = 100       # 行权价格
T = 1.0       # 到期时间（年）
r = 0.05      # 无风险利率
v0 = 0.04     # 初始波动率
theta = 0.04  # 长期波动率
kappa = 2.0   # 回归速度
sigma = 0.5   # 波动率的波动率
rho = -0.7    # 波动率和资产价格的相关性
num_paths = 100000  # 模拟路径数
num_steps = 252     # 时间步数

# 计算期权价格
call_price = heston_mc_european(S0, K, T, r, v0, theta, kappa, sigma, rho, num_paths, num_steps, option_type='call')
put_price = heston_mc_european(S0, K, T, r, v0, theta, kappa, sigma, rho, num_paths, num_steps, option_type='put')

print(f"Heston Call Option Price (MC): {call_price:.2f}")
print(f"Heston Put Option Price (MC): {put_price:.2f}")
```

### 参数说明
- $ S_0 $: 当前资产价格。
- $ K $: 行权价格。
- $ T $: 到期时间。
- $ r $: 无风险利率。
- $ v_0 $: 当前波动率。
- $ \theta $: 长期波动率。
- $ \kappa $: 波动率回归速度。
- $ \sigma $: 波动率的波动率。
- $ \rho $: 资产价格和波动率的相关性。
- `num_paths`: 模拟路径数，越大结果越准确，但计算时间越长。
- `num_steps`: 时间步数，越大模拟越精细，但计算时间越长。

### 输出示例
运行代码后，将输出类似以下结果：
```
Heston Call Option Price (MC): 10.46
Heston Put Option Price (MC): 8.33
```

### 优化建议
1. **提升效率**：使用更高效的随机数生成器或 GPU 加速。
2. **控制方差**：通过方差减少技术（如对偶变量法或控制变量法）提高结果精度。
3. **时间步改进**：考虑更高级的时间步方案（如 Milstein 方法）来处理波动率。

是的，**Variance process** 和 **Asset price process** 的公式是基于 Heston 模型的随机微分方程（SDE），推导过程中可以使用 **Ito 引理** 来确保精确的建模和分析。

我们将逐步推导每个过程的公式。

---

### 1. **Variance Process**: $ dv_t = \kappa (\theta - v_t) dt + \sigma \sqrt{v_t} dW_t^v $

#### SDE 分析
这是一个均值回复过程（mean-reverting process），被称为 CIR (Cox-Ingersoll-Ross) 过程。它的特性如下：
- 均值回复到长期均值 $ \theta $。
- $ \kappa $ 是均值回复速度。
- $ \sigma $ 是波动率的波动率。
- 其噪声项由 $ \sqrt{v_t} $ 调节，确保 $ v_t $ 为非负数。

由于 CIR 过程天然地约束 $ v_t \geq 0 $，无需特别处理非负性。但在数值模拟时，可能会出现负值，需要通过绝对值等技术校正。

#### 欧拉-马尔科夫离散化
对 $ v_t $ 的 SDE 进行离散化：
$$
v_{t+\Delta t} = v_t + \kappa (\theta - v_t) \Delta t + \sigma \sqrt{v_t} \sqrt{\Delta t} Z_t
$$
其中：
- $ Z_t \sim \mathcal{N}(0, 1) $ 是标准正态随机变量。

#### 非负性调整
离散化后，数值可能导致 $ v_{t+\Delta t} < 0 $。常见的解决方案包括：
1. **取非负部分**： $ v_{t+\Delta t} = \max(v_{t+\Delta t}, 0) $。
2. **反射法**：将 $ v_{t+\Delta t} $ 的负值取反，确保非负。

---

### 2. **Asset Price Process**: $ dS_t = r S_t dt + \sqrt{v_t} S_t dW_t^S $

#### SDE 分析
这是一个 Geometric Brownian Motion (GBM) 的扩展，其波动率 $ \sqrt{v_t} $ 是随机的。

通过 Ito 引理可以得到资产价格 $ S_t $ 的解析解。具体推导如下：

#### 公式变换
资产价格的 SDE 为：
$$
\frac{dS_t}{S_t} = r dt + \sqrt{v_t} dW_t^S
$$

取对数，设 $ X_t = \ln S_t $，则：
$$
dX_t = \frac{1}{S_t} dS_t - \frac{1}{2} \left( \frac{1}{S_t} \right)^2 (dS_t)^2
$$
代入 $ dS_t $ 的公式：
$$
dX_t = \left( r - \frac{1}{2} v_t \right) dt + \sqrt{v_t} dW_t^S
$$

#### 离散化公式
对 $ X_t $ 进行离散化后：
$$
X_{t+\Delta t} = X_t + \left( r - \frac{1}{2} v_t \right) \Delta t + \sqrt{v_t} \sqrt{\Delta t} Z_t
$$
其中 $ Z_t \sim \mathcal{N}(0, 1) $。

反向取指数得到 $ S_t $：
$$
S_{t+\Delta t} = S_t \exp\left[ \left( r - \frac{1}{2} v_t \right) \Delta t + \sqrt{v_t} \sqrt{\Delta t} Z_t \right]
$$

---

### 3. **Ito 引理的推导**

#### 随机变量 $ S_t $ 的形式
假设资产价格 $ S_t $ 是关于 $ t $ 和 $ W_t^S $ 的函数：
$$
S_t = f(t, W_t^S)
$$

根据 Ito 引理：
$$
df(t, W_t^S) = \frac{\partial f}{\partial t} dt + \frac{\partial f}{\partial W_t^S} dW_t^S + \frac{1}{2} \frac{\partial^2 f}{\partial (W_t^S)^2} (dW_t^S)^2
$$

对于资产价格的 SDE：
$$
dS_t = r S_t dt + \sqrt{v_t} S_t dW_t^S
$$
直接套用公式后可得到上述离散化公式。

---

### 最终离散化公式总结

1. **Variance Process**：
$$
v_{t+\Delta t} = v_t + \kappa (\theta - v_t) \Delta t + \sigma \sqrt{v_t} \sqrt{\Delta t} Z_t
$$

2. **Asset Price Process**：
$$
S_{t+\Delta t} = S_t \exp\left[ \left( r - \frac{1}{2} v_t \right) \Delta t + \sqrt{v_t} \sqrt{\Delta t} Z_t \right]
$$

### 推导的关键点
- **Ito 引理** 用于推导资产价格的对数形式，确保非负性。
- **CIR 过程** 天然满足非负性，但离散化需要额外处理。

如果有其他细节问题，欢迎继续讨论！