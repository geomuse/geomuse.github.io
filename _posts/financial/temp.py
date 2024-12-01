import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib import style
style.use('ggplot')
# Parameters for the European options
S = np.linspace(10, 200, 500)  # Range of underlying prices
K = 100  # Strike price
T = 1    # Time to maturity (in years)
r = 0.05  # Risk-free rate
sigma = 0.2  # Volatility

# Black-Scholes formula components
def d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

# Delta for call and put options
def delta_call(S, K, T, r, sigma):
    return norm.cdf(d1(S, K, T, r, sigma))

def delta_put(S, K, T, r, sigma):
    return norm.cdf(d1(S, K, T, r, sigma)) - 1

# Calculate deltas
delta_c = delta_call(S, K, T, r, sigma)
delta_p = delta_put(S, K, T, r, sigma)

# Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(S, delta_c, label='Delta (Call Option)', lw=2)
# plt.plot(S, delta_p, label='Delta (Put Option)', lw=2, linestyle='--')
# plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
# plt.title('Delta of Call and Put Options')
# plt.xlabel('Underlying Price (S)')
# plt.ylabel('Delta')
# plt.legend()
# plt.grid(alpha=0.3)
# plt.show()

# Gamma calculation
def gamma(S, K, T, r, sigma):
    d1_val = d1(S, K, T, r, sigma)
    return norm.pdf(d1_val) / (S * sigma * np.sqrt(T))

# Calculate gamma
gamma_values = gamma(S, K, T, r, sigma)

# Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(S, gamma_values, label='Gamma', lw=2, color='purple')
# plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
# plt.title('Gamma of Call and Put Options')
# plt.xlabel('Underlying Price (S)')
# plt.ylabel('Gamma')
# plt.legend()
# plt.grid(alpha=0.3)
# plt.show()

# Theta calculation
def theta_call(S, K, T, r, sigma):
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    first_term = -(S * norm.pdf(d1_val) * sigma) / (2 * np.sqrt(T))
    second_term = -r * K * np.exp(-r * T) * norm.cdf(d2_val)
    return first_term + second_term

def theta_put(S, K, T, r, sigma):
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    first_term = -(S * norm.pdf(d1_val) * sigma) / (2 * np.sqrt(T))
    second_term = r * K * np.exp(-r * T) * norm.cdf(-d2_val)
    return first_term + second_term

# Calculate theta for call and put options
theta_c = theta_call(S, K, T, r, sigma)
theta_p = theta_put(S, K, T, r, sigma)

# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(S, theta_c, label='Theta (Call Option)', lw=2, color='blue')
# plt.plot(S, theta_p, label='Theta (Put Option)', lw=2, linestyle='--', color='red')
# plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
# plt.title('Theta of Call and Put Options')
# plt.xlabel('Underlying Price (S)')
# plt.ylabel('Theta')
# plt.legend()
# plt.grid(alpha=0.3)
# plt.show()

# Vega calculation
def vega(S, K, T, r, sigma):
    d1_val = d1(S, K, T, r, sigma)
    return S * norm.pdf(d1_val) * np.sqrt(T) / 100  # Scaled by 100 for percentage representation

# Calculate vega
vega_values = vega(S, K, T, r, sigma)

# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(S, vega_values, label='Vega', lw=2, color='green')
# plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
# plt.title('Vega of Call and Put Options')
# plt.xlabel('Underlying Price (S)')
# plt.ylabel('Vega')
# plt.legend()
# plt.grid(alpha=0.3)
# plt.show()

# Rho calculation
def rho_call(S, K, T, r, sigma):
    d2_val = d2(S, K, T, r, sigma)
    return K * T * np.exp(-r * T) * norm.cdf(d2_val) / 100  # Scaled by 100 for percentage representation

def rho_put(S, K, T, r, sigma):
    d2_val = d2(S, K, T, r, sigma)
    return -K * T * np.exp(-r * T) * norm.cdf(-d2_val) / 100  # Scaled by 100 for percentage representation

# Calculate rho for call and put options
rho_c = rho_call(S, K, T, r, sigma)
rho_p = rho_put(S, K, T, r, sigma)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(S, rho_c, label='Rho (Call Option)', lw=2, color='blue')
plt.plot(S, rho_p, label='Rho (Put Option)', lw=2, linestyle='--', color='orange')
plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
plt.title('Rho of Call and Put Options')
plt.xlabel('Underlying Price (S)')
plt.ylabel('Rho')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
