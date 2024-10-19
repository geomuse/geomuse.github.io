import numpy as np

def barrier_option_price(S0, K, T, r, sigma, B, M, N, option_type='call', barrier_type='up-and-out'):
    # S0: 初始资产价格
    # K: 执行价格
    # T: 到期时间
    # r: 无风险利率
    # sigma: 波动率
    # B: 障碍价格
    # M: 模拟路径数
    # N: 每个路径的时间步数
    # option_type: 期权类型 'call' 或 'put'
    # barrier_type: 障碍类型 'up-and-out', 'down-and-out', 'up-and-in', 'down-and-in'

    dt = T / N  # 每步的时间长度
    discount_factor = np.exp(-r * T)  # 折现因子
    total_payoff = 0.0

    # 生成 M 条资产价格路径
    for _ in range(M):
        S = S0
        hit_barrier = False
        for _ in range(N):
            Z = np.random.normal(0, 1)
            S = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
            
            # 判断是否触发障碍
            if barrier_type == 'up-and-out' and S >= B:
                hit_barrier = True
                break
            elif barrier_type == 'down-and-out' and S <= B:
                hit_barrier = True
                break
            elif barrier_type == 'up-and-in' and S >= B:
                hit_barrier = True
            elif barrier_type == 'down-and-in' and S <= B:
                hit_barrier = True

        # 根据期权类型和障碍类型计算期末的支付
        if barrier_type in ['up-and-out', 'down-and-out']:
            if not hit_barrier:
                if option_type == 'call':
                    payoff = max(S - K, 0)
                elif option_type == 'put':
                    payoff = max(K - S, 0)
                total_payoff += payoff
        elif barrier_type in ['up-and-in', 'down-and-in']:
            if hit_barrier:
                if option_type == 'call':
                    payoff = max(S - K, 0)
                elif option_type == 'put':
                    payoff = max(K - S, 0)
                total_payoff += payoff

    # 计算期权价格
    option_price = discount_factor * (total_payoff / M)
    return option_price

# 示例参数
S0 = 100  # 初始价格
K = 105   # 执行价格
T = 1.0   # 到期时间（1年）
r = 0.05  # 无风险利率
sigma = 0.2  # 波动率
B = 110   # 障碍价格
M = 10000  # 模拟路径数
N = 100    # 时间步数

price = barrier_option_price(S0, K, T, r, sigma, B, M, N, option_type='call', barrier_type='up-and-out')
price_ = barrier_option_price(S0, K, T, r, sigma, B, M, N, option_type='call', barrier_type='up-and-in')
print(f'Barrier Option Price: {price + price_}')

import matplotlib.pyplot as pt

c = 2.0
s0 = np.linspace(0,200,200)
# profit = []

# for s in s0 :
#     profit.append(barrier_option_price(s, K, T, r, sigma, B, M, N, option_type='call', barrier_type='up-and-out')-c)

profit = [barrier_option_price(s, K, T, r, sigma, B, M, N, option_type='call', barrier_type='up-and-out')-c for s in s0]

pt.plot(s0,profit,'-',color='blue')
pt.xlabel('underlying price')
pt.ylabel('profit')
pt.axvline(x = K,color='r',linestyle='--', linewidth=2)
pt.show()