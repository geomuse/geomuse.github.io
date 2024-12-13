import numpy as np
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    """
    Black-Scholes formula for European call option pricing.

    Parameters:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate
        sigma (float): Volatility of the stock

    Returns:
        float: Call option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# Example Parameters
S = 100  # Current stock price
K = 100  # Strike price
T = 1    # Time to maturity (1 year)
r = 0.05  # Risk-free rate
sigma = 0.2  # Volatility

call_price = black_scholes_call(S, K, T, r, sigma)
print(f"European Call Option Price: {call_price:.2f}")

def monte_carlo_call_price(S, K, T, r, sigma, num_simulations=10000):
    """
    Monte Carlo simulation for European call option pricing.

    Parameters:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate
        sigma (float): Volatility of the stock
        num_simulations (int): Number of Monte Carlo simulations

    Returns:
        float: Call option price
    """
    dt = T
    ST = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal(size=num_simulations))
    payoff = np.maximum(ST - K, 0)
    call_price = np.exp(-r * T) * np.mean(payoff)
    return call_price

# Example Parameters
num_simulations = 100000
call_price_mc = monte_carlo_call_price(S, K, T, r, sigma, num_simulations)
print(f"European Call Option Price (Monte Carlo): {call_price_mc:.2f}")

import matplotlib.pyplot as plt

# Simulate Monte Carlo paths
def simulate_paths(S, T, r, sigma, num_simulations=1000, n_steps=252):
    dt = T / n_steps
    paths = np.zeros((num_simulations, n_steps + 1))
    paths[:, 0] = S
    for t in range(1, n_steps + 1):
        z = np.random.normal(size=num_simulations)
        paths[:, t] = paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return paths

paths = simulate_paths(S, T, r, sigma)
plt.plot(paths.T, lw=0.5, alpha=0.7)
plt.title("Simulated Stock Price Paths")
plt.xlabel("Time Steps")
plt.ylabel("Stock Price")
plt.show()
