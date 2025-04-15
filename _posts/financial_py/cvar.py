from pymongo import MongoClient
import pandas as pd 
import matplotlib.pyplot as pt
import numpy as np
from matplotlib import style
style.use('ggplot')

# 连接到 MongoDB（替换为实际的连接字符串）
client = MongoClient("mongodb://localhost:27017/")
# 选择数据库（例如 "mydatabase"）
db = client["stock_data"]
collection = db["AAPL"]

# 查询数据，例如：读取所有数据
results = collection.find()
print(data := pd.DataFrame(results))

close = data["Close"]

def calculate_var_(returns, confidence_level=0.95):
    """
    计算给定收益序列的 VaR 值
    参数:
        returns: numpy 数组，表示投资组合的收益率（或损失）
        confidence_level: 置信水平（默认为 0.95，即 95%）
    返回:
        VaR 值，对应 (1 - confidence_level) 分位数
    """
    # 例如，对于95%置信水平，计算 5% 分位数
    return np.percentile(returns, (1 - confidence_level) * 100)

def calculate_cvar_(returns, confidence_level=0.95):
    """
    计算给定收益序列的 CVaR 值（条件 VaR，也称预期损失）
    参数:
        returns: numpy 数组，表示投资组合的收益率（或损失）
        confidence_level: 置信水平（默认为 0.95）
    返回:
        CVaR 值，即在损失超过 VaR 部分的平均损失
    """
    var_val = calculate_var_(returns, confidence_level)
    # 取所有低于或等于 VaR 的收益率（即损失较大的情况），求均值
    return returns[returns <= var_val].mean()

def calculate_var(losses, beta=0.95):
    """
    计算 VaR
    参数:
      losses: numpy 数组，表示样本损失数据（假设损失为正数）
      beta: 置信水平（例如 0.95 表示 95%）
    返回:
      VaR 值，即 (1-beta)*100 分位数
    """
    # 计算 (1-beta) 分位数，作为 VaR 的估计值
    alpha = np.percentile(losses, (1 - beta) * 100)
    return alpha

def calculate_cvar(losses, beta=0.95):
    """
    根据公式 CVaR = alpha + 1/(1-beta) * E[(L - alpha)_+] 计算条件 VaR（CVaR）
    参数:
      losses: numpy 数组，表示样本损失数据
      beta: 置信水平（例如 0.95 表示 95%）
    返回:
      CVaR 值
    """
    # 先计算 VaR
    alpha = calculate_var(losses, beta)
    
    # 计算每个样本中超出 VaR 部分的损失
    # (L - alpha)_+ = max(L - alpha, 0)
    excess_losses = np.maximum(losses - alpha, 0)
    
    # 这里用样本均值来近似期望 E[(L-alpha)_+]
    mean_excess = np.mean(excess_losses)
    
    # 根据公式计算 CVaR
    cvar = alpha + mean_excess / (1 - beta)
    return cvar

if __name__ == "__main__":

    # 示例：生成模拟损失数据并计算 VaR 与 CVaR
    # 模拟 1000 个损失数据，假设损失服从均值为 5，标准差为 1 的正态分布
    losses =  close.pct_change().dropna()

    beta = 0.95  # 置信水平
    var_value = calculate_var(losses, beta)
    cvar_value = calculate_cvar(losses, beta)

    var_value = calculate_var_(losses, confidence_level=0.95)
    cvar_value = calculate_cvar_(losses, confidence_level=0.95)

    print(f"VaR (95%): {var_value:4f}")
    print(f"CVaR (95%): {cvar_value:4f}")
    print(f"VaR ({beta*100}%): {var_value:.4f}")
    print(f"CVaR ({beta*100}%): {cvar_value:.4f}")
