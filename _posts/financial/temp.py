#%%
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

#%%
import matplotlib.pyplot as pt
pt.plot(np.arange(0, len(returns)), returns)

#%%

import numpy as np
import matplotlib.pyplot as pt

def geometric_brownian_motion(S0, mu, sigma, T, dt):
    N = int(T / dt)
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size=N) 
    W = np.cumsum(W) * np.sqrt(dt)  # Wiener process
    S = S0 * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * W)
    return t, S

# 参数
S0 = 100      # 初始价格
mu = 0.05     # 预期收益率
sigma = 0.2   # 波动率
T = 1         # 模拟时间1年
dt = 0.01     # 时间步长

# 生成数据并绘图
t, S = geometric_brownian_motion(S0, mu, sigma, T, dt)
pt.plot(t, S)
pt.xlabel("Time (Years)")
pt.ylabel("Stock Price")
pt.title("Geometric Brownian Motion")
pt.show()

# %%