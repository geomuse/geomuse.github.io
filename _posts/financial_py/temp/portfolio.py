from pymongo import MongoClient
import pandas as pd 
import matplotlib.pyplot as pt
import numpy as np
from matplotlib import style
from scipy.optimize import minimize
style.use('ggplot')

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

# 计算投资组合的年化波动率
def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))

def portfolio_return(weights, returns):
    return np.sum(returns.mean() * weights) * 252

# 定义优化目标：最小化波动率
def min_volatility(weights):
    return portfolio_volatility(weights, cov_matrix)

if __name__ == '__main__' :

    AAPL_return = AAPL["Close"].pct_change().dropna()
    GOOGL_return = GOOGL["Close"].pct_change().dropna()
    MSFT_return = MSFT["Close"].pct_change().dropna()

    returns = pd.DataFrame({
        "AAPL" : AAPL_return,
        "GOOGL" : GOOGL_return,
        "MSFT" : MSFT_return
        })
    cov_matrix = returns.cov()

    num_assets = 3
    init_guess = np.array(num_assets * [1. / num_assets])

    # 设置权重的边界和约束条件
    bounds = tuple((0, 1) for asset in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

    # 进行优化
    optimized = minimize(min_volatility, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    # 输出优化结果
    print("Optimized Weights:", optimized.x)
    print("Minimum Volatility:", optimized.fun)

    
    # 定义蒙特卡洛模拟的次数
    num_portfolios = 10000

    # 存储结果
    results = np.zeros((3, num_portfolios))

    for i in range(num_portfolios):
        # 生成随机权重
        weights = np.random.random(3)
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