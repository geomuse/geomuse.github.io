---
layout: post
title:  fe bootstrapping
date:   2024-10-10 11:24:29 +0800
categories: 
    - financial 
    - python
---

<script type="text/javascript" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

利率期限结构（Term Structure of Interest Rates）描述的是不同到期时间的债券或金融工具的收益率（利率）之间的关系，通常以收益率曲线的形式表示。这种关系可以通过多个数学模型进行描述，主要用于理解市场对未来利率的预期、评估债券价格、管理利率风险以及进行投资组合管理。下面介绍几种常见的利率期限结构模型：

### 1. 传统的利率期限结构模型
这些模型通常基于市场观察到的数据来拟合收益率曲线，它们包括：

- **Nelson-Siegel模型**：一个半参数模型，用于拟合利率曲线。它的公式相对简单，具有平移、倾斜和曲率三个参数，可以很好地描述利率曲线的不同形状。
- **Svensson扩展模型**：基于Nelson-Siegel模型，增加了更多参数以进一步增强对不同曲率的拟合能力，能够更灵活地捕捉不同期限利率之间的变化。

### 2. 无套利模型
这些模型基于无套利的假设，来描述整个期限结构。它们通常用于复杂的金融工具定价，比如期权或利率衍生品。这些模型包括：

- **Vasicek模型**：一个单因子短期利率模型。它假设短期利率服从均值回复的随机过程，因此可以描述利率的波动性和长期均衡水平。然而，Vasicek模型允许负利率。
  
- **Cox-Ingersoll-Ross（CIR）模型**：改进了Vasicek模型，使短期利率始终为正。它同样假设均值回复，且利率的波动性与利率水平正相关，这意味着在高利率情况下波动会加大。

- **Ho-Lee模型**：该模型是第一个基于无套利理论的随机利率模型，假设短期利率服从布朗运动。它的最大特点是可以根据市场数据来校准模型参数，以生成符合市场的收益率曲线。

### 3. 随机利率模型
这些模型假设利率是由一个或多个随机过程驱动的，这些过程的动态通常比较复杂，但可以更好地捕捉市场的波动性。

- **Hull-White模型**：基于Vasicek模型的扩展，可以通过动态调整来适应当前的收益率曲线。Hull-White模型增加了灵活性，能够精确地贴合当前的市场数据。

- **Black-Karasinski模型**：类似于Hull-White模型，但是假设利率的对数值服从均值回复过程。这可以让模型更适用于模拟实际中观察到的利率分布形状。

### 4. HJM 框架（Heath-Jarrow-Morton）
- **HJM模型**：HJM（Heath-Jarrow-Morton）模型直接描述远期利率的演化过程，而不是短期利率。它的一个显著优点是能够产生与实际市场数据匹配的任意形状的收益率曲线。然而，HJM模型通常比较复杂，不容易用于直接计算利率期权的价格。

### 5. LIBOR市场模型
- **LIBOR市场模型（LMM）**：也称为Brace-Gatarek-Musiela（BGM）模型，通常用于定价基于LIBOR的衍生工具。它假设每一个期限的利率都是由独立的随机过程驱动的，这种方法可以更好地模拟市场实际中的即期利率行为。

### 6. 马尔科夫跳跃模型
- **CIR模型的跳跃扩展**：在经典的CIR模型基础上增加了跳跃扩展，可以模拟突发性的利率变化。这些模型特别适合描述利率市场中可能出现的非常规波动，如政策变化等。

### 总结
利率期限结构模型在金融学中是非常重要的工具，能够用于描述不同期限利率之间的关系和变化趋势。选择合适的模型取决于具体的应用场景：

- 对于一般性的利率曲线拟合和市场数据的表示，**Nelson-Siegel** 和 **Svensson** 是不错的选择。
- 对于定价和管理利率风险，尤其是考虑无套利的条件下，**Vasicek**、**CIR**、**Hull-White** 是常用的短期利率模型。
- 对于更复杂的衍生品定价，**HJM** 和 **LIBOR市场模型** 能更好地适应市场的波动和复杂性。

以下是一些利率期限结构模型的 Python 实现，涵盖了 Vasicek、CIR、Hull-White 等常见模型。

### Vasicek model
Vasicek 模型假设利率遵循均值回复的随机过程，数学表达式为：
$$
dr_t = a(b - r_t)dt + \sigma dW_t
$$

```python
import numpy as np
import matplotlib.pyplot as plt

# Vasicek 模型参数
a = 0.1  # 均值回复速度
b = 0.05  # 长期均值
sigma = 0.02  # 波动率
r0 = 0.03  # 初始利率
T = 1.0  # 模拟总时间（年）
dt = 0.01  # 时间步长
N = int(T / dt)  # 总步数

# 初始化利率路径
rates = np.zeros(N)
rates[0] = r0

# 使用 Euler 方法模拟短期利率路径
for t in range(1, N):
    dr = a * (b - rates[t-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
    rates[t] = rates[t-1] + dr

# 绘图
time = np.linspace(0, T, N)
plt.plot(time, rates)
plt.xlabel('Time (years)')
plt.ylabel('Interest Rate')
plt.title('Vasicek Model Interest Rate Path')
plt.show()
```

### CIR model
CIR 模型在 Vasicek 模型的基础上添加了利率对波动的依赖，表达式为：
$$
dr_t = a(b - r_t)dt + \sigma \sqrt{r_t} dW_t
$$

```python
import numpy as np
import matplotlib.pyplot as plt

# CIR 模型参数
a = 0.1  # 均值回复速度
b = 0.05  # 长期均值
sigma = 0.02  # 波动率
r0 = 0.03  # 初始利率
T = 1.0  # 模拟总时间（年）
dt = 0.01  # 时间步长
N = int(T / dt)  # 总步数

# 初始化利率路径
rates = np.zeros(N)
rates[0] = r0

# 使用 Euler 方法模拟短期利率路径
for t in range(1, N):
    dr = a * (b - rates[t-1]) * dt + sigma * np.sqrt(rates[t-1]) * np.sqrt(dt) * np.random.normal()
    rates[t] = rates[t-1] + dr
    rates[t] = max(rates[t], 0)  # 保证利率非负

# 绘图
time = np.linspace(0, T, N)
plt.plot(time, rates)
plt.xlabel('Time (years)')
plt.ylabel('Interest Rate')
plt.title('CIR Model Interest Rate Path')
plt.show()
```

### Hull-White model
Hull-White 模型是一种扩展的 Vasicek 模型，其数学表示为：

$$
dr_t = (\theta(t) - a r_t)dt + \sigma dW_t
$$

其中 $$ \theta(t) $$ 是时间相关的参数，可以根据市场数据进行校准。

```python
import numpy as np
import matplotlib.pyplot as plt

# Hull-White 模型参数
a = 0.1  # 均值回复速度
sigma = 0.02  # 波动率
r0 = 0.03  # 初始利率
T = 1.0  # 模拟总时间（年）
dt = 0.01  # 时间步长
N = int(T / dt)  # 总步数

# 时间相关的均值函数
def theta(t):
    return 0.05  # 这里假设是常数，可以根据需要调整

# 初始化利率路径
rates = np.zeros(N)
rates[0] = r0

# 使用 Euler 方法模拟短期利率路径
for t in range(1, N):
    dr = (theta(t * dt) - a * rates[t-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
    rates[t] = rates[t-1] + dr

# 绘图
time = np.linspace(0, T, N)
plt.plot(time, rates)
plt.xlabel('Time (years)')
plt.ylabel('Interest Rate')
plt.title('Hull-White Model Interest Rate Path')
plt.show()
```

### Svensson model

```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
from scipy.optimize import minimize

# 定义Nelson-Siegel-Svensson模型函数
def svensson(tau, beta0, beta1, beta2, beta3, lambda1, lambda2):
    term1 = beta0
    term2 = beta1 * ((1 - np.exp(-tau / lambda1)) / (tau / lambda1))
    term3 = beta2 * (((1 - np.exp(-tau / lambda1)) / (tau / lambda1)) - np.exp(-tau / lambda1))
    term4 = beta3 * (((1 - np.exp(-tau / lambda2)) / (tau / lambda2)) - np.exp(-tau / lambda2))
    return term1 + term2 + term3 + term4

# 定义目标函数
def objective_function(params, tau, y):
    beta0, beta1, beta2, beta3, lambda1, lambda2 = params
    y_model = svensson(tau, beta0, beta1, beta2, beta3, lambda1, lambda2)
    error = y - y_model
    return np.sum(error ** 2)

# 市场数据（期限：年，收益率：%）
data = pd.DataFrame({
    'tau': [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30],
    'yield': [1.5, 1.7, 1.9, 2.1, 2.3, 2.7, 3.0, 3.2, 3.5, 3.7]
})

tau = data['tau'].values
y = data['yield'].values

# 初始参数猜测
initial_params = [3.0, -1.0, 1.0, 0.5, 1.0, 3.0]

# 参数约束和边界
bounds = (
    (None, None),  # beta0
    (None, None),  # beta1
    (None, None),  # beta2
    (None, None),  # beta3
    (0.0001, None),  # lambda1
    (0.0001, None)   # lambda2
)

# 优化参数
result = minimize(objective_function, initial_params, args=(tau, y), bounds=bounds, method='L-BFGS-B')

beta0_opt, beta1_opt, beta2_opt, beta3_opt, lambda1_opt, lambda2_opt = result.x

print("优化后的参数：")
print(f"beta0 = {beta0_opt}")
print(f"beta1 = {beta1_opt}")
print(f"beta2 = {beta2_opt}")
print(f"beta3 = {beta3_opt}")
print(f"lambda1 = {lambda1_opt}")
print(f"lambda2 = {lambda2_opt}")

# 计算拟合后的收益率
tau_fit = np.linspace(0.1, 30, 300)
y_fit = svensson(tau_fit, beta0_opt, beta1_opt, beta2_opt, beta3_opt, lambda1_opt, lambda2_opt)

# 绘制结果
pt.figure(figsize=(10, 6))
pt.plot(tau, y, 'o', label='市场收益率')
pt.plot(tau_fit, y_fit, '-', label='NS Svensson模型拟合')
pt.xlabel('期限（年）')
pt.ylabel('收益率（%）')
pt.title('Nelson-Siegel-Svensson模型拟合利率曲线')
pt.legend()
pt.grid(True)
pt.show()
```