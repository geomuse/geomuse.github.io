
```py
import yfinance as yf
import numpy as np
import pandas as pd

# 股票池（举例）
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
data = yf.download(tickers, start="2020-01-01", end="2024-12-31")['Adj Close']

# 计算对数收益率
returns = np.log(data / data.shift(1)).dropna()
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252
```

最大化 Sharpe Ratio

```py
from scipy.optimize import minimize

def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    returns = np.dot(weights, mean_returns)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (returns - risk_free_rate) / std
    return returns, std, sharpe

def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]

def get_max_sharpe_weights(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    result = minimize(negative_sharpe, num_assets*[1./num_assets], args=(mean_returns, cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x
```

最小化风险（波动率）

```py
def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def get_min_vol_weights(cov_matrix):
    num_assets = len(cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    result = minimize(portfolio_volatility, num_assets*[1./num_assets], args=(cov_matrix,),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

```