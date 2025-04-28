#%%
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from pymongo import MongoClient

# 连接到 MongoDB（替换为实际的连接字符串）
client = MongoClient("mongodb://localhost:27017/")

# 选择数据库（例如 "mydatabase"）
db = client["stock_data"]

# 选择集合（例如 "mycollection"）
AAPL = db["AAPL"]
GOOGL = db['GOOGL']
MSFT = db['MSFT']

# 查询数据，例如：读取所有数据
AAPL = AAPL.find()
GOOGL = GOOGL.find()
MSFT = MSFT.find()

# 输出查询结果
print(AAPL := pd.DataFrame(AAPL))
print(GOOGL := pd.DataFrame(GOOGL))
print(MSFT := pd.DataFrame(MSFT))

df = pd.DataFrame({
    'AAPL' : AAPL['Close'] , 
    'GOOGL' : GOOGL['Close'] ,
    'MSFT' : MSFT['Close']
})
tickers = ['AAPL',"GOOGL","MSFT"]
returns = df.pct_change().dropna()

# 年化收益和协方差矩阵
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

# 投资组合性能指标
def portfolio_performance(weights):
    ret = np.dot(weights, mean_returns)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = ret / vol
    return ret, vol, sharpe

# 目标函数：最小方差
def minimize_volatility(weights):
    return portfolio_performance(weights)[1]

# 目标函数：最大Sharpe Ratio (负号用于最小化)
def negative_sharpe_ratio(weights):
    return -portfolio_performance(weights)[2]

# 约束：权重总和为1
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(len(tickers)))
initial_guess = len(tickers) * [1. / len(tickers)]

# 最小方差组合
min_vol_result = minimize(minimize_volatility, initial_guess,
                          method='SLSQP', bounds=bounds, constraints=constraints)

# 最大Sharpe Ratio组合
max_sharpe_result = minimize(negative_sharpe_ratio, initial_guess,
                             method='SLSQP', bounds=bounds, constraints=constraints)

# 输出结果
print("最小方差组合权重:")
print(dict(zip(tickers, min_vol_result.x.round(4))))
ret, vol, sharpe = portfolio_performance(min_vol_result.x)
print(f"年化收益: {ret:.2%}, 年化波动率: {vol:.2%}, Sharpe Ratio: {sharpe:.2f}\n")

print("最大Sharpe Ratio组合权重:")
print(dict(zip(tickers, max_sharpe_result.x.round(4))))
ret, vol, sharpe = portfolio_performance(max_sharpe_result.x)
print(f"年化收益: {ret:.2%}, 年化波动率: {vol:.2%}, Sharpe Ratio: {sharpe:.2f}")