import numpy as np
from scipy.stats import norm

def black_scholes_european_option(S0, K, r, T, sigma, option_type="call"):
    """
    Black-Scholes model for European Option pricing.

    Parameters:
        S0: float - Initial stock price.
        K: float - Strike price.
        r: float - Risk-free interest rate.
        T: float - Time to maturity (in years).
        sigma: float - Volatility of the underlying asset.
        option_type: str - "call" or "put".

    Returns:
        float: Price of the European option.
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    return price

# Example usage
S0 = 100         # Initial stock price
K = 100          # Strike price
r = 0.05         # Risk-free rate
T = 1            # Time to maturity (1 year)
sigma = 0.2      # Volatility
barrier = 110    # Barrier level


# European call option using Black-Scholes
bs_call_price = black_scholes_european_option(S0, K, r, T, sigma, option_type="call")
print(f"European Call Option Price (Black-Scholes): {bs_call_price:.4f}")

# European put option using Black-Scholes
bs_put_price = black_scholes_european_option(S0, K, r, T, sigma, option_type="put")
print(f"European Put Option Price (Black-Scholes): {bs_put_price:.4f}")
