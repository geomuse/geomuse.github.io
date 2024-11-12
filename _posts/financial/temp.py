import numpy as np
from scipy.stats import norm

def black_scholes_greeks(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta = norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * norm.cdf(d2) if option_type == "call"
             else -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2))
    rho = K * T * np.exp(-r * T) * norm.cdf(d2) if option_type == "call" else -K * T * np.exp(-r * T) * norm.cdf(-d2)
    
    return {"Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta, "Rho": rho}

def monte_carlo_greeks(S, K, T, r, sigma, simulations=10000):
    Z = np.random.standard_normal(simulations)
    S_T = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(S_T - K, 0)
    option_price = np.exp(-r * T) * np.mean(payoff)
    
    dS = 0.01
    S_T_up = (S + dS) * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    payoff_up = np.maximum(S_T_up - K, 0)
    option_price_up = np.exp(-r * T) * np.mean(payoff_up)
    delta = (option_price_up - option_price) / dS
    
    d_sigma = 0.01
    S_T_vega = S * np.exp((r - 0.5 * (sigma + d_sigma) ** 2) * T + (sigma + d_sigma) * np.sqrt(T) * Z)
    payoff_vega = np.maximum(S_T_vega - K, 0)
    option_price_vega = np.exp(-r * T) * np.mean(payoff_vega)
    vega = (option_price_vega - option_price) / d_sigma
    
    return {"Delta": delta, "Vega": vega}

def finite_difference_gamma(S, K, T, r, sigma):
    dS = 0.01
    price_up = black_scholes_greeks(S + dS, K, T, r, sigma)["Delta"]
    price_down = black_scholes_greeks(S - dS, K, T, r, sigma)["Delta"]
    gamma = (price_up - price_down) / (2 * dS)
    
    return gamma

import numpy as np

def finite_difference_option_price(S, K, T, r, sigma, M, N, option_type="call"):
    dt = T / M
    dS = S / N
    grid = np.zeros((N + 1, M + 1))
    
    # 资产价格离散值
    S_vals = np.linspace(0, S * 2, N + 1)
    
    # 终值条件和边界条件
    if option_type == "call":
        grid[:, -1] = np.maximum(S_vals - K, 0)
        grid[-1, :] = S_vals[-1] - K * np.exp(-r * np.linspace(0, T, M + 1))
    else:
        grid[:, -1] = np.maximum(K - S_vals, 0)
        grid[0, :] = K * np.exp(-r * np.linspace(0, T, M + 1))
    
    # 系数计算
    alpha = 0.5 * dt * ((sigma ** 2) * (np.arange(N) ** 2) - r * np.arange(N))
    beta = 1 - dt * ((sigma ** 2) * (np.arange(N) ** 2) + r)
    gamma = 0.5 * dt * ((sigma ** 2) * (np.arange(N) ** 2) + r * np.arange(N))

    for j in range(M - 1, -1, -1):
        for i in range(1, N):
            grid[i, j] = alpha[i] * grid[i - 1, j + 1] + beta[i] * grid[i, j + 1] + gamma[i] * grid[i + 1, j + 1]

    # 返回期权初始价格
    return np.interp(S, S_vals, grid[:, 0])

# Greeks 计算函数
def finite_difference_greeks(S, K, T, r, sigma, M, N, option_type="call"):
    epsilon = 1e-4  # 微小变化值

    # 计算初始期权价格
    price = finite_difference_option_price(S, K, T, r, sigma, M, N, option_type)

    # Delta
    price_up = finite_difference_option_price(S + epsilon, K, T, r, sigma, M, N, option_type)
    delta = (price_up - price) / epsilon

    # Gamma
    price_down = finite_difference_option_price(S - epsilon, K, T, r, sigma, M, N, option_type)
    gamma = (price_up - 2 * price + price_down) / (epsilon ** 2)

    # Theta
    price_theta = finite_difference_option_price(S, K, T - epsilon, r, sigma, M, N, option_type)
    theta = (price_theta - price) / epsilon

    # Vega
    price_vega = finite_difference_option_price(S, K, T, r, sigma + epsilon, M, N, option_type)
    vega = (price_vega - price) / epsilon

    # Rho
    price_rho = finite_difference_option_price(S, K, T, r + epsilon, sigma, M, N, option_type)
    rho = (price_rho - price) / epsilon

    return {"Delta": delta, "Gamma": gamma, "Theta": theta, "Vega": vega, "Rho": rho}

# 参数设置
S = 100
K = 100  
T = 1    
r = 0.05 
sigma = 0.2 
M = 1000    # 时间步数
N = 100     # 空间步数

# 计算 Greeks
greeks = finite_difference_greeks(S, K, T, r, sigma, M, N, option_type="call")
print("Greeks:", greeks)

greeks = black_scholes_greeks(S, K, T, r, sigma, option_type="call")
print("Greeks:", greeks)

greeks_monte_carlo = monte_carlo_greeks(S, K, T, r, sigma)
print("Monte Carlo Greeks:", greeks_monte_carlo)

gamma_fd = finite_difference_gamma(S, K, T, r, sigma)
print("Finite Difference Gamma:", gamma_fd)