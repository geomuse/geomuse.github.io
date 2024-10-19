import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as pt
from matplotlib import style
style.use('ggplot')

def european_option_price(S0, K, T, r, sigma, option_type='call'):
    """
    计算欧式期权价格的 Black-Scholes 模型
    :param S0: 标的资产价格
    :param K: 行权价格
    :param T: 到期时间
    :param r: 无风险利率
    :param sigma: 波动率
    :param option_type: 期权类型 'call' 或 'put'
    :return: 期权价格
    """
    
    # 计算 d1 和 d2
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        # 欧式看涨期权价格
        price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        # 欧式看跌期权价格
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    
    return price

# 示例参数
S0 = 100  # 当前标的资产价格
K = 105   # 行权价格
T = 1.0   # 到期时间（1年）
r = 0.05  # 无风险利率
sigma = 0.2  # 波动率

# 计算欧式看涨期权价格
call_price = european_option_price(S0, K, T, r, sigma, option_type='call')
print(f"欧式看涨期权价格: {call_price}")

# 计算欧式看跌期权价格
put_price = european_option_price(S0, K, T, r, sigma, option_type='put')
print(f"欧式看跌期权价格: {put_price}")

# pi = price - c
c = 2.0
s0 = np.linspace(0,200,200)
profit = european_option_price(s0, K, T, r, sigma, option_type='call') - c 
# print(x)

pt.plot(s0,profit,'-',color='blue')
pt.xlabel('underlying price')
pt.ylabel('profit')
pt.axvline(x = K,color='r',linestyle='--', linewidth=2)
pt.show()