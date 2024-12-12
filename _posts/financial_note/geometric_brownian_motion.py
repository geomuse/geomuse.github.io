import numpy as np
import matplotlib.pyplot as pt
from matplotlib import style
style.use('ggplot')

def brownian_motion(n=1000, T=1):
    dt = T / n
    W = np.zeros(n)
    for i in range(1, n):
        W[i] = W[i-1] + np.sqrt(dt) * np.random.normal()
    return W

n = 1000
T = 1
W = brownian_motion(n, T)
t = np.linspace(0, T, n)

pt.plot(t, W)
pt.title("Brownian Motion")
pt.xlabel("Time")
pt.ylabel("W(t)")
pt.show()

def geometric_brownian_motion(S0, mu, sigma, n=1000, T=1):
    dt = T / n
    W = brownian_motion(n, T)
    t = np.linspace(0, T, n)
    S = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)
    return S

S0 = 100  # Initial price
mu = 0.1  # Drift
sigma = 0.2  # Volatility
S = geometric_brownian_motion(S0, mu, sigma)

pt.plot(t, S)
pt.title("Geometric Brownian Motion")
pt.xlabel("Time")
pt.ylabel("S(t)")
pt.show()

def ornstein_uhlenbeck(x0, mu, theta, sigma, n=1000, T=1):
    dt = T / n
    X = np.zeros(n)
    X[0] = x0
    for i in range(1, n):
        dW = np.sqrt(dt) * np.random.normal()
        X[i] = X[i-1] + theta * (mu - X[i-1]) * dt + sigma * dW
    return X

x0 = 0.5
mu = 0.0
theta = 0.7
sigma = 0.2
X = ornstein_uhlenbeck(x0, mu, theta, sigma)

pt.plot(t, X)
pt.title("Ornstein-Uhlenbeck Process")
pt.xlabel("Time")
pt.ylabel("X(t)")
pt.show()

def cir_model(r0, mu, theta, sigma, n=1000, T=1):
    dt = T / n
    r = np.zeros(n)
    r[0] = r0
    for i in range(1, n):
        dW = np.sqrt(dt) * np.random.normal()
        r[i] = r[i-1] + theta * (mu - r[i-1]) * dt + sigma * np.sqrt(max(r[i-1], 0)) * dW
    return r

r0 = 0.03
mu = 0.05
theta = 0.1
sigma = 0.02
r = cir_model(r0, mu, theta, sigma)

pt.plot(t, r)
pt.title("CIR Model")
pt.xlabel("Time")
pt.ylabel("r(t)")
pt.show()

def jump_diffusion(S0, mu, sigma, lam, mu_j, sigma_j, n=1000, T=1):
    dt = T / n
    S = np.zeros(n)
    S[0] = S0
    for i in range(1, n):
        dW = np.sqrt(dt) * np.random.normal()
        jump = np.random.poisson(lam * dt)
        S[i] = S[i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW) * np.exp(jump * np.random.normal(mu_j, sigma_j))
    return S

S0 = 100
mu = 0.1
sigma = 0.2
lam = 0.1
mu_j = 0.02
sigma_j = 0.05
S_jump = jump_diffusion(S0, mu, sigma, lam, mu_j, sigma_j)

pt.plot(t, S_jump)
pt.title("Jump Diffusion Model")
pt.xlabel("Time")
pt.ylabel("S(t)")
pt.show()
