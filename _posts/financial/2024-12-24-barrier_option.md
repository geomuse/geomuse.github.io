---
layout: post
title : barrier option monte carlo method (updated) 
date : 2024-12-24 11:24:29 +0800
categories: 
    - financial
    - option
    - python
---

以下是基于蒙特卡洛模拟的 Python 代码，用于计算障碍期权（Barrier Option）的价格。

```py
import numpy as np

def monte_carlo_barrier_option(S0, K, r, T, sigma, barrier, option_type="call", barrier_type="up-and-out", n_simulations=100000, n_steps=100):
    """
    Monte Carlo simulation for Barrier Option pricing.

    Parameters:
        S0: float - Initial stock price.
        K: float - Strike price.
        r: float - Risk-free interest rate.
        T: float - Time to maturity (in years).
        sigma: float - Volatility of the underlying asset.
        barrier: float - Barrier level.
        option_type: str - "call" or "put".
        barrier_type: str - "up-and-out", "up-and-in", "down-and-out", or "down-and-in".
        n_simulations: int - Number of Monte Carlo simulations.
        n_steps: int - Number of time steps.

    Returns:
        float: Estimated price of the barrier option.
    """
    dt = T / n_steps
    discount_factor = np.exp(-r * T)

    # Simulate asset paths
    S = np.zeros((n_simulations, n_steps + 1))
    S[:, 0] = S0

    for t in range(1, n_steps + 1):
        Z = np.random.standard_normal(n_simulations)
        S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

    # Check barrier conditions
    if barrier_type == "up-and-out":
        barrier_breach = np.any(S >= barrier, axis=1)
        valid_paths = ~barrier_breach
    elif barrier_type == "down-and-out":
        barrier_breach = np.any(S <= barrier, axis=1)
        valid_paths = ~barrier_breach
    elif barrier_type == "up-and-in":
        barrier_breach = np.any(S >= barrier, axis=1)
        valid_paths = barrier_breach
    elif barrier_type == "down-and-in":
        barrier_breach = np.any(S <= barrier, axis=1)
        valid_paths = barrier_breach
    else:
        raise ValueError("Invalid barrier type. Use 'up-and-out', 'up-and-in', 'down-and-out', or 'down-and-in'.")

    # Option payoff calculation
    if option_type == "call":
        payoff = np.maximum(S[:, -1] - K, 0)
    elif option_type == "put":
        payoff = np.maximum(K - S[:, -1], 0)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    # Apply valid paths condition
    payoff[~valid_paths] = 0

    # Discount back to present value
    option_price = discount_factor * np.mean(payoff)

    return option_price

# Example usage
S0 = 100         # Initial stock price
K = 100          # Strike price
r = 0.05         # Risk-free rate
T = 1            # Time to maturity (1 year)
sigma = 0.2      # Volatility
barrier = 110    # Barrier level

# Up-and-out call option
up_and_out_price = monte_carlo_barrier_option(S0, K, r, T, sigma, barrier, option_type="call", barrier_type="up-and-out")
print(f"Up-and-Out Call Option Price: {up_and_out_price:.4f}")
up_and_out_price = monte_carlo_barrier_option(S0, K, r, T, sigma, barrier, option_type="put", barrier_type="up-and-out")
print(f"Up-and-Out Put Option Price: {up_and_out_price:.4f}")

# Down-and-out call option
down_and_out_price = monte_carlo_barrier_option(S0, K, r, T, sigma, barrier, option_type="call", barrier_type="down-and-out")
print(f"Down-and-Out Call Option Price: {down_and_out_price:.4f}")
down_and_out_price = monte_carlo_barrier_option(S0, K, r, T, sigma, barrier, option_type="put", barrier_type="down-and-out")
print(f"Down-and-Out Put Option Price: {down_and_out_price:.4f}")

# Up-and-in call option
up_and_in_price = monte_carlo_barrier_option(S0, K, r, T, sigma, barrier, option_type="call", barrier_type="up-and-in")
print(f"Up-and-In Call Option Price: {up_and_in_price:.4f}")
up_and_in_price = monte_carlo_barrier_option(S0, K, r, T, sigma, barrier, option_type="put", barrier_type="up-and-in")
print(f"Up-and-In Put Option Price: {up_and_in_price:.4f}")

# Down-and-in call option
down_and_in_price = monte_carlo_barrier_option(S0, K, r, T, sigma, barrier, option_type="call", barrier_type="down-and-in")
print(f"Down-and-In Call Option Price: {down_and_in_price:.4f}")
down_and_in_price = monte_carlo_barrier_option(S0, K, r, T, sigma, barrier, option_type="put", barrier_type="down-and-in")
print(f"Down-and-In Put Option Price: {down_and_in_price:.4f}")
```