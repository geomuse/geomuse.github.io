---
layout: post
title : 投资组合与多目标优化
date : 2024-12-22 11:24:29 +0800
categories: 
    - financial
    - portfolio
    - python
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

### **现代投资组合理论（Modern Portfolio Theory, MPT）**

1. **收益**：
   - 组合的期望收益：  
     $$
     E(R_p) = \sum_{i=1}^{n} w_i E(R_i)
     $$  
     - $E(R_p)$：投资组合的期望收益  
     - $w_i$：资产 $i$ 的权重  
     - $E(R_i)$：资产 $i$ 的期望收益  

2. **风险**：
   - 组合的方差（衡量风险）：  
     $$
     \sigma_p^2 = \sum_{i=1}^{n} \sum_{j=1}^{n} w_i w_j \text{Cov}(R_i, R_j)
     $$  
     - $ \sigma_p^2 $：组合的总风险  
     - $\text{Cov}(R_i, R_j)$：资产 $i$ 和 $j$ 的收益协方差  

3. **协方差与相关系数**：
   - 协方差衡量两资产收益的相关性：
     $$
     \text{Cov}(R_i, R_j) = \rho_{i,j} \cdot \sigma_i \cdot \sigma_j
     $$  
     - $\rho_{i,j}$：资产 $i$ 和 $j$ 的相关系数  
     - 分散化效果越强，相关系数 $\rho_{i,j}$ 越低。

<!-- #### **有效前沿（Efficient Frontier）**：
- **定义**：在给定风险水平下，收益最高的投资组合，或在给定收益目标下，风险最低的投资组合。
- **图解**：有效前沿是一条曲线，位于风险-收益空间中，展示最优组合。 -->

<!-- --- -->

<!-- ### **3. 投资组合优化的核心指标** -->

<!-- 1. **期望收益率**：
   衡量投资组合未来的潜在回报。
   
2. **风险（方差/标准差）**：
   衡量投资组合的不确定性或波动性。 -->

4. **夏普比率（Sharpe Ratio）**：
   衡量单位风险下的超额收益。
   $$
   \text{Sharpe Ratio} = \frac{E(R_p) - R_f}{\sigma_p}
   $$  
   - $R_f$：无风险收益率  
   - $E(R_p)$：组合期望收益率  
   - $\sigma_p$：组合标准差

<!-- 4. **贝塔系数（Beta）**：
   衡量投资组合对市场波动的敏感性。 -->

![Image Description](/assets/images/pic.png)

![Image Description](/assets/images/pic1.png)

![Image Description](/assets/images/pic2.png)

![Image Description](/assets/images/pic3.png)


```py
# 标准的寻找最小波动率带来的投资组合权重
Optimized Weights:  [0.30085163 0.13994133 0.28229909 0.27690795]

Expected Portfolio Return:  0.2591958475852417
Minimum Volatility:  0.3007599529050759
Sharpe Ratio: 0.7620557370468108

---

# alpha * (-portfolio_return) + (1 - alpha) * portfolio_volatility）
# 多目标优化下的投资组合权重
Optimal Weights: [0.58245443 0.00000000 0.0107118  0.40683377]

Expected Portfolio Return: 0.2889
Portfolio Risk: 0.3128
Sharpe Ratio: 0.8276
```

### 附录

```py
import yfinance as yf
import pandas as pd
import numpy as np
from matplotlib import style
from scipy.optimize import minimize
import matplotlib.pyplot as pt
style.use('ggplot')

class markowiz :

    # 计算投资组合的预期收益和年化波动率
    def portfolio_return(self, weights, returns):
        return np.sum(returns.mean() * weights) * 252

    # 计算投资组合的年化波动率
    def portfolio_volatility(self, weights, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))

    # 定义优化目标：最小化波动率
    def min_volatility(self, weights, cov_matrix):
        return self.portfolio_volatility(weights, cov_matrix)

    # 多目标优化函数
    def objective_function(self, weights, returns, cov_matrix, risk_free_rate, alpha=0.5):
        # portfolio_return = np.dot(weights, mean_returns)  # 组合收益
        portfolio_return = self.portfolio_return(weights,returns)
        
        # portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))  # 组合风险
        portfolio_risk = self.portfolio_volatility(weights,cov_matrix)
        
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk  # 夏普比率

        # 目标：平衡收益和风险，alpha 控制权重
        return alpha * (-portfolio_return) + (1 - alpha) * portfolio_risk

    # 权重和约束
    def weight_sum_constraint(self, weights):
        return np.sum(weights) - 1

if __name__ == '__main__' :

    # 定义股票代码
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']

    # 下载股票数据
    data = yf.download(tickers, start="2020-01-01", end="2024-01-01")['Adj Close']

    risk_free_rate = 0.03 
    
    # 计算每日收益率
    returns = data.pct_change().dropna()

    # 计算协方差矩阵
    cov_matrix = returns.cov()

    # 初始猜测：假设均匀分配
    num_assets = len(tickers)
    init_guess = np.array(num_assets * [1. / num_assets])

    # 设置权重的边界和约束条件
    bounds = tuple((0, 1) for asset in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

    optimized = minimize(markowiz().min_volatility, init_guess, args=(cov_matrix),method='SLSQP', bounds=bounds, constraints=constraints)

    print("Optimized Weights: ", optimized.x)
    print("\nExpected Portfolio Return: ", portfolio_return := markowiz().portfolio_return(optimized.x,returns))
    print("Minimum Volatility: ", optimized.fun)
    sharpe_ratio = (portfolio_return - risk_free_rate) / optimized.fun
    print(f"Sharpe Ratio: {sharpe_ratio}\n")
    # 权重非负约束
    bounds = [(0, 1) for _ in range(len(tickers))]

    # 优化
    constraints = {'type': 'eq', 'fun': markowiz().weight_sum_constraint}
    alpha = 0.5  # 平衡系数
    result = minimize(
        markowiz().objective_function,
        optimized.x,
        args=(returns, cov_matrix, risk_free_rate, alpha),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    # 输出结果
    optimal_weights = result.x
    portfolio_return = markowiz().portfolio_return(optimal_weights,returns)
    portfolio_risk = markowiz().portfolio_volatility(optimal_weights,cov_matrix)  
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk

    print(f"Optimal Weights: {optimal_weights}")
    print(f"\nExpected Portfolio Return: {portfolio_return:.4f}")
    print(f"Portfolio Risk: {portfolio_risk:.4f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
```