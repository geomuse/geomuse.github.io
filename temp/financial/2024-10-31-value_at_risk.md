---
layout: post
title : value at risk
date : 2024-10-31 11:24:29 +0800
categories: 
    - financial
    - risk
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

VaR 是用于量化投资组合或单个资产在特定置信水平下的最大潜在损失。计算方法主要包括历史法、方差-协方差法和蒙特卡洛模拟法。

查看`chatgpt`

https://chatgpt.com/share/671de82b-38d0-800f-8f3b-5685b617eddb/continue


使用蒙特卡洛模拟法计算 VaR

```py
import numpy as np
import pandas as pd
import yfinance as yf

# 下载股票历史数据
ticker = "AAPL"  # 例如苹果公司
data = yf.download(ticker, start="2022-01-01", end="2024-01-01")
returns = data['Adj Close'].pct_change().dropna()

# 蒙特卡洛模拟 VaR
np.random.seed(0)
simulations = 10000  # 模拟次数
confidence_level = 0.95
simulated_returns = np.random.normal(returns.mean(), returns.std(), simulations)
VaR_mc = np.percentile(simulated_returns, (1 - confidence_level) * 100)

print(f"蒙特卡洛模拟 VaR at {confidence_level*100}%: {VaR_mc * 100:.2f}%")
```

历史模拟法 (Historical Simulation)

通过使用资产或组合的历史收益率数据，计算 VaR。

```py
import numpy as np
import pandas as pd
import yfinance as yf

# 下载股票历史数据
ticker = "AAPL"
data = yf.download(ticker, start="2022-01-01", end="2024-01-01")
returns = data['Adj Close'].pct_change().dropna()

# 置信水平
confidence_level = 0.95

# 历史模拟 VaR
VaR_hist = np.percentile(returns, (1 - confidence_level) * 100)
print(f"历史模拟 VaR at {confidence_level*100}%: {VaR_hist * 100:.2f}%")
```

方差-协方差法 (Variance-Covariance Method)

$$VaR = z * \sigma-\mu$$

假设资产收益率符合正态分布，通过均值和标准差计算 VaR。

```py
from scipy.stats import norm

# 均值和标准差
mean_return = returns.mean()
std_dev = returns.std()

# 置信水平下的 z 值
confidence_level = 0.95
z_score = norm.ppf(1 - confidence_level)

# 方差-协方差法 VaR
VaR_cov = z_score * std_dev - mean_return
print(f"方差协方差法 VaR at {confidence_level*100}%: {VaR_cov * 100:.2f}%")
```

$$VaR = \Delta * \sigma * z$$

专门应用于带有衍生品（如期权）投资组合的 VaR 计算，假设投资组合的风险因子是线性组合。

```py
# 设置 Delta 值 (假设对价格变动有较高敏感度)
delta = 1

# Delta-Normal VaR
VaR_delta_normal = delta * std_dev * z_score
print(f"Delta-Normal VaR at {confidence_level*100}%: {VaR_delta_normal * 100:.2f}%")
```

$$CVaR$$

也称为 Expected Shortfall，是 VaR 的一种扩展，表示在超出 VaR 的情况下的平均损失。

```py
# 条件性 VaR (CVaR)
CVaR = simulated_returns[simulated_returns <= VaR_mc].mean()
print(f"条件性 VaR at {confidence_level*100}%: {CVaR * 100:.2f}%")
```

### Portfolio VaR 组合 VaR

当计算投资组合的 VaR 时，可以采用历史模拟法、方差协方差法或蒙特卡洛模拟，并根据每只股票的权重计算组合的 VaR。以下是用方差协方差法的组合 VaR 计算示例：

```py
tickers = ["AAPL", "MSFT", "GOOGL"]  # 股票代码
weights = [0.3, 0.4, 0.3]  # 投资组合中每只股票的权重

# 获取每只股票的历史数据
data = yf.download(tickers, start="2022-01-01", end="2024-01-01")['Adj Close']
returns = data.pct_change().dropna()

# 计算协方差矩阵
cov_matrix = returns.cov()

# 组合标准差
portfolio_std_dev = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))

# 组合 VaR
portfolio_VaR = z_score * portfolio_std_dev
print(f"组合 VaR at {confidence_level*100}%: {portfolio_VaR * 100:.2f}%")
```

### Component VaR（成分 VaR）

成分 VaR 主要用于分析每只资产在组合风险中的贡献。

```py
# 计算组合的成分 VaR
component_VaR = weights * (z_score * np.sqrt(np.diag(cov_matrix)))
component_contributions = component_VaR / portfolio_VaR
print("成分 VaR 贡献：", component_contributions)
```

成分VaR和边际VaR是用于风险分解和风险管理的两种重要方法，特别是在多资产组合中。这两者与总体VaR有关，通过它们可以更好地理解每个资产或资产类别对组合总体VaR的贡献。

### 1. 总体VaR (Total VaR)
总体VaR是衡量整个投资组合在特定置信水平和持有期内可能面临的最大损失。例如，一个95%置信水平的VaR表示在最坏的5%的情况下，组合可能损失的最大金额。

### 2. 成分VaR (Component VaR)
成分VaR表示每个资产（或资产类别）对投资组合总体VaR的贡献。它通过分解组合的VaR，帮助投资者了解每个资产对组合总体风险的具体影响。

**计算成分VaR**：
成分VaR可以通过以下公式计算：
$$
\text{成分VaR}_i = \frac{\partial \text{组合VaR}}{\partial w_i} \cdot w_i
$$
其中 \( \frac{\partial \text{组合VaR}}{\partial w_i} \) 是边际VaR，\( w_i \) 是资产i在组合中的权重。成分VaR反映了资产i对组合VaR的加权贡献。

成分VaR的求和等于组合总体VaR：
$$
\text{组合VaR} = \sum_{i} \text{成分VaR}_i
$$

### 3. 边际VaR (Marginal VaR)
边际VaR是每个资产在当前组合权重下对总体VaR的边际贡献。具体来说，边际VaR表示如果增加该资产的权重，组合VaR将增加的风险量。它衡量每个资产的“额外风险贡献”。

**计算边际VaR**：
边际VaR的公式为：
$$
\text{边际VaR}_i = \frac{\partial \text{组合VaR}}{\partial w_i}
$$

边际VaR可以通过计算组合的VaR对资产权重的偏导数获得。

### 成分VaR、边际VaR与总体VaR的关系
- 总体VaR是整个组合在特定置信水平下的潜在最大损失。
- 成分VaR通过每个资产的边际VaR和其组合权重衡量该资产对总体VaR的具体贡献。
- 边际VaR代表单个资产对组合总体VaR的“边际”影响，通常用于判断在组合中增加或减少某一资产权重的风险变化。

这种关系帮助投资者在分散风险时进行组合优化，例如减少高成分VaR的资产权重，从而降低组合的总体VaR。