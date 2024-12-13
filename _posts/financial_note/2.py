import numpy as np

def heston_mc_european(S0, K, T, r, v0, theta, kappa, sigma, rho, num_paths, num_steps, option_type='call'):
    """
    Heston model Monte Carlo simulation for European option pricing.
    
    Parameters:
        S0: float - Initial stock price
        K: float - Strike price
        T: float - Time to maturity (in years)
        r: float - Risk-free interest rate
        v0: float - Initial variance
        theta: float - Long-term mean variance
        kappa: float - Mean reversion speed
        sigma: float - Volatility of variance
        rho: float - Correlation between stock and variance
        num_paths: int - Number of Monte Carlo paths
        num_steps: int - Number of time steps
        option_type: str - 'call' or 'put'
    
    Returns:
        float - Option price
    """
    dt = T / num_steps  # Time step
    S = np.zeros((num_steps + 1, num_paths))
    v = np.zeros((num_steps + 1, num_paths))
    S[0] = S0
    v[0] = v0

    # Simulate paths
    for t in range(1, num_steps + 1):
        # Generate correlated random numbers
        Z1 = np.random.normal(0, 1, num_paths)
        Z2 = np.random.normal(0, 1, num_paths)
        W1 = Z1
        W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2

        # Variance process (ensure non-negativity)
        v[t] = np.abs(v[t - 1] + kappa * (theta - v[t - 1]) * dt + sigma * np.sqrt(v[t - 1] * dt) * W2)

        # Asset price process
        S[t] = S[t - 1] * np.exp((r - 0.5 * v[t - 1]) * dt + np.sqrt(v[t - 1] * dt) * W1)

    # Calculate option payoff
    if option_type == 'call':
        payoff = np.maximum(S[-1] - K, 0)
    elif option_type == 'put':
        payoff = np.maximum(K - S[-1], 0)
    else:
        raise ValueError("Invalid option type. Choose 'call' or 'put'.")

    # Discount payoff to present value
    option_price = np.exp(-r * T) * np.mean(payoff)
    return option_price

# 参数设置
S0 = 100      # 初始资产价格
K = 100       # 行权价格
T = 1.0       # 到期时间（年）
r = 0.05      # 无风险利率
v0 = 0.04     # 初始波动率
theta = 0.04  # 长期波动率
kappa = 2.0   # 回归速度
sigma = 0.5   # 波动率的波动率
rho = -0.7    # 波动率和资产价格的相关性
num_paths = 100000  # 模拟路径数
num_steps = 252     # 时间步数

# 计算期权价格
call_price = heston_mc_european(S0, K, T, r, v0, theta, kappa, sigma, rho, num_paths, num_steps, option_type='call')
put_price = heston_mc_european(S0, K, T, r, v0, theta, kappa, sigma, rho, num_paths, num_steps, option_type='put')

print(f"Heston Call Option Price (MC): {call_price:.2f}")
print(f"Heston Put Option Price (MC): {put_price:.2f}")
