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