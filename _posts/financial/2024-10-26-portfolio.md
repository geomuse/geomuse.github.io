---
layout: post
title : markowiz and portfolio management
date : 2024-10-26 11:24:29 +0800
categories: 
    - financial
    - portfolio
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

下载数据

```py
import yfinance as yf
import pandas as pd
import numpy as np
from matplotlib import style
from scipy.optimize import minimize
import matplotlib.pyplot as pt
style.use('ggplot')
# 定义股票代码
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']

# 下载股票数据
data = yf.download(tickers, start="2020-01-01", end="2024-01-01")['Adj Close']

# 计算每日收益率
returns = data.pct_change().dropna()
```

```py
# 计算协方差矩阵
cov_matrix = returns.cov()

# 计算投资组合的年化波动率
def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))

# 定义优化目标：最小化波动率
def min_volatility(weights):
    return portfolio_volatility(weights, cov_matrix)

# 初始猜测：假设均匀分配
num_assets = len(tickers)
init_guess = np.array(num_assets * [1. / num_assets])

# 设置权重的边界和约束条件
bounds = tuple((0, 1) for asset in range(num_assets))
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

# 进行优化
optimized = minimize(min_volatility, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)

# 输出优化结果
print("Optimized Weights:", optimized.x)
print("Minimum Volatility:", optimized.fun)
```

```py
# 计算投资组合的预期收益和年化波动率
def portfolio_return(weights, returns):
    return np.sum(returns.mean() * weights) * 252

def portfolio_performance(weights, returns, cov_matrix):
    port_return = portfolio_return(weights, returns)
    port_volatility = portfolio_volatility(weights, cov_matrix)
    return port_return, port_volatility

opt_return, opt_volatility = portfolio_performance(optimized.x, returns, cov_matrix)

print("Expected Portfolio Return:", opt_return)
print("Expected Portfolio Volatility:", opt_volatility)
```

```py
# 定义蒙特卡洛模拟的次数
num_portfolios = 10000

# 存储结果
results = np.zeros((3, num_portfolios))

for i in range(num_portfolios):
    # 生成随机权重
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)
    
    # 计算投资组合的收益率和波动率
    port_return = portfolio_return(weights, returns)
    port_volatility = portfolio_volatility(weights, cov_matrix)
    
    # 存储结果
    results[0,i] = port_return
    results[1,i] = port_volatility
    results[2,i] = port_return / port_volatility  # 夏普比率

# 获取最高夏普比率的组合
max_sharpe_idx = np.argmax(results[2])

max_sharpe = results[2,max_sharpe_idx]

sdp, rp = results[1,max_sharpe_idx], results[0,max_sharpe_idx]

# 获取最小波动率的组合
min_vol_idx = np.argmin(results[1])
sdp_min, rp_min = results[1,min_vol_idx], results[0,min_vol_idx]

pt.figure(figsize=(10, 7))
pt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='YlGnBu', marker='x',alpha=0.7)
pt.title('Monte Carlo Simulation of Portfolio Optimization')
pt.xlabel('Volatility')
pt.ylabel('Return')
pt.colorbar(label='Sharpe Ratio')

# 标记最高夏普比率组合和最小波动率组合
pt.scatter(sdp, rp, marker='*', color='r', s=200, label='Maximum Sharpe Ratio')
pt.scatter(sdp_min, rp_min, marker='*', color='g', s=200, label='Minimum Volatility')
pt.legend(labelspacing=0.8)
pt.show()

print(f'returns : {sdp} , volatility : {rp} , sharpe ratio : {max_sharpe}')
```

![Image Description](/assets/images/pic.png)

```py
#%% cal

# 定义无风险利率
risk_free_rate = 0.02

# 计算CAL斜率 (市场投资组合的夏普比率)
slope_cal = (rp - risk_free_rate) / sdp

# 生成CAL数据点
x = np.linspace(0, 0.5, 100)
cal_y = risk_free_rate + slope_cal * x

# 绘制CAL
pt.figure(figsize=(10, 7))
pt.plot(x, cal_y, linestyle='--', color='orange', label='Capital Market Line (CAL)')

pt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='YlGnBu', marker='x',alpha=0.7)
pt.scatter(sdp, rp, marker='*', color='r', s=200, label='Market Portfolio (Max Sharpe)')

pt.title('Capital Market Line (CAL) and Monte Carlo Simulation')
pt.xlabel('Volatility')
pt.ylabel('Return')
pt.xlim([0.2,0.4])
# pt.colorbar(label='Sharpe Ratio')
pt.legend(labelspacing=0.8)
pt.show()
```

$$cal = \frac{E(R)-R_f}{\sigma_p}\sigma_c+R_f$$

![Image Description](/assets/images/pic1.png)


```py
# max_sharpe = results[2,max_sharpe_idx]
# print(max_sharpe)

#%% 风险厌恶系数
 
# 定义风险厌恶系数
A = 3.0

# 计算无差异曲线
def utility_curve(A, return_p, volatility_p):
    return return_p - 0.5 * A * volatility_p**2

# 生成无差异曲线
volatility_range = np.linspace(0.01, 0.5, 100)
utility_levels = [0.1, 0.15, 0.2]  # 假设三个不同的效用值

pt.figure(figsize=(10, 7))

# 绘制无差异曲线
for level in utility_levels:
    utility_curve_y = [utility_curve(A, level, vol) for vol in volatility_range]
    pt.plot(volatility_range, utility_curve_y, linestyle='--', label=f'Indifference Curve (Utility={level})')

# 绘制CAL和蒙特卡洛模拟结果
pt.plot(x, cal_y, linestyle='-', color='orange', label='Capital Market Line (CAL)')
pt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='YlGnBu', marker='o', alpha=0.3)
pt.scatter(sdp, rp, marker='*', color='r', s=200, label='Market Portfolio (Max Sharpe)')
pt.title('Indifference Curves, Capital Market Line (CAL), and Monte Carlo Simulation')
pt.xlabel('Volatility')
pt.ylabel('Return')
pt.colorbar(label='Sharpe Ratio')
pt.legend(labelspacing=0.8)

pt.show()
```

![Image Description](/assets/images/pic2.png)


```py
import numpy as np

# 目标函数：负夏普比率（我们希望最大化夏普比率）
def objective_function(weights, returns, cov_matrix, risk_free_rate):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return -sharpe_ratio

# 约束条件：权重和为1
def constraint(weights):
    return np.sum(weights) - 1

# 初始参数
num_assets = len(tickers)
initial_weights = np.array(num_assets * [1. / num_assets])
initial_temperature = 1.0
cooling_rate = 0.99
min_temperature = 1e-8
max_iterations = 1000

# 随机生成一个新解
def generate_neighbor(weights):
    new_weights = weights + np.random.uniform(-0.1, 0.1, len(weights))
    new_weights = np.abs(new_weights)
    new_weights /= np.sum(new_weights)
    return new_weights

# 模拟退火过程
def simulated_annealing(returns, cov_matrix, risk_free_rate, initial_weights, initial_temperature, cooling_rate, min_temperature, max_iterations):
    current_weights = initial_weights
    current_score = objective_function(current_weights, returns, cov_matrix, risk_free_rate)
    
    best_weights = current_weights
    best_score = current_score
    
    temperature = initial_temperature
    
    for i in range(max_iterations):
        new_weights = generate_neighbor(current_weights)
        new_score = objective_function(new_weights, returns, cov_matrix, risk_free_rate)
        
        # 接受新解的概率
        if new_score < current_score or np.exp((current_score - new_score) / temperature) > np.random.rand():
            current_weights = new_weights
            current_score = new_score
            
            # 如果找到更好的解，更新最佳解
            if new_score < best_score:
                best_weights = new_weights
                best_score = new_score
        
        # 降温
        temperature *= cooling_rate
        
        # 检查温度是否达到最小值
        if temperature < min_temperature:
            break
    
    return best_weights, best_score

# 运行模拟退火算法
best_weights, best_score = simulated_annealing(returns, cov_matrix, risk_free_rate, initial_weights, initial_temperature, cooling_rate, min_temperature, max_iterations)

print("Best Weights Found:", best_weights)
print("Best Sharpe Ratio:", -best_score)

# 绘制蒙特卡洛模拟结果和模拟退火结果
pt.figure(figsize=(10, 7))
pt.scatter(results[1,:], results[0,:], cmap='YlGnBu', marker='x', alpha=0.2)
pt.scatter(sdp, rp, marker='x', color='r', s=100, label='Market Portfolio (Max Sharpe)')
pt.scatter(np.sqrt(np.dot(best_weights.T, np.dot(cov_matrix * 252, best_weights))), 
            np.sum(returns.mean() * best_weights) * 252, 
            marker='x', color='purple', s=100, label='Simulated Annealing Portfolio')
pt.scatter(sdp_min, rp_min, marker='x', color='g', s=100, label='Minimum Volatility')

pt.title('Simulated Annealing Optimization vs. Monte Carlo Simulation')
pt.xlabel('Volatility')
pt.ylabel('Return')
# pt.colorbar(label='Sharpe Ratio')
pt.legend(labelspacing=0.8)

pt.show()
```

![Image Description](/assets/images/pic3.png)

```py
#%% 实现了投资组合优化，旨在最大化夏普比率和最小化投资组合的波动率
# 假设无风险利率为 0
risk_free_rate = 0.0

def sharpe_ratio(weights, returns, cov_matrix):
    port_return = portfolio_return(weights, returns)
    port_volatility = portfolio_volatility(weights, cov_matrix)
    return (port_return - risk_free_rate) / port_volatility

def neg_sharpe_ratio(weights, returns, cov_matrix):
    # 最大化夏普比率等价于最小化负夏普比率
    return -sharpe_ratio(weights, returns, cov_matrix)

# 进行优化
optimized_sharpe = minimize(neg_sharpe_ratio, init_guess, args=(returns, cov_matrix), method='SLSQP', bounds=bounds, constraints=constraints)

# 输出优化结果
print("Optimized Weights for Max Sharpe:", optimized_sharpe.x)
print("Maximum Sharpe Ratio:", -optimized_sharpe.fun)

# 假设我们希望每只股票的权重不超过30%，不低于5%
bounds = tuple((0.05, 0.3) for asset in range(num_assets))

# 重新进行优化
optimized_with_bounds = minimize(min_volatility, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)

# 输出优化结果
print("Optimized Weights with Bounds:", optimized_with_bounds.x)
print("Minimum Volatility with Bounds:", optimized_with_bounds.fun)
```

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

#### **有效前沿（Efficient Frontier）**：
- **定义**：在给定风险水平下，收益最高的投资组合，或在给定收益目标下，风险最低的投资组合。
- **图解**：有效前沿是一条曲线，位于风险-收益空间中，展示最优组合。

---

### **3. 投资组合优化的核心指标**

1. **期望收益率**：
   衡量投资组合未来的潜在回报。
   
2. **风险（方差/标准差）**：
   衡量投资组合的不确定性或波动性。

3. **夏普比率（Sharpe Ratio）**：
   衡量单位风险下的超额收益。
   $$
   \text{Sharpe Ratio} = \frac{E(R_p) - R_f}{\sigma_p}
   $$  
   - $R_f$：无风险收益率  
   - $E(R_p)$：组合期望收益率  
   - $\sigma_p$：组合标准差

4. **贝塔系数（Beta）**：
   衡量投资组合对市场波动的敏感性。