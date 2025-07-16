---
layout: post
title:  volatility smile 
date:   2024-10-28 11:24:29 +0800
categories: 
    - financial
    - volatility
---

<script>
  MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']]
    }
  };
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

除了对市场的波动率进行插值后得到相对应的非标准期限的隐含波动率外 , 还有3个随机波动率模型可以针对隐含波动率

目前市场上最流行的模型是 `bates model` 且在代码中只针对期权价格,但可以用`black scholes model`对于隐含波动率的求根

https://chatgpt.com/share/67335c76-f038-800f-aec7-246c6888a3be

### bates model

```py
import numpy as np
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize, brentq
from scipy.stats import norm

class bates_model :
    # Bates 模型特征函数（包含跳跃项）
    def bates_characteristic_function(self,u, params, S0, r, q, T):
        kappa, theta, sigma, rho, v0, lambd, muJ, sigmaJ = params
        D1 = np.sqrt((rho * sigma * 1j * u - kappa) ** 2 + sigma ** 2 * (1j * u + u ** 2))
        G = (kappa - rho * sigma * 1j * u - D1) / (kappa - rho * sigma * 1j * u + D1)
        eDT = np.exp(-D1 * T)
        C = kappa * theta / sigma ** 2 * ((kappa - rho * sigma * 1j * u - D1) * T - 2 * np.log((1 - G * eDT) / (1 - G)))
        D = (kappa - rho * sigma * 1j * u - D1) / sigma ** 2 * ((1 - eDT) / (1 - G * eDT))
        M = np.exp(1j * u * (np.log(S0) + (r - q - lambd * (np.exp(muJ + 0.5 * sigmaJ ** 2) - 1)) * T))
        J = np.exp(lambd * T * (np.exp(1j * u * muJ - 0.5 * u ** 2 * sigmaJ ** 2) - 1))
        return M * np.exp(C + D * v0) * J

    # 期权价格积分（与 Heston 类似）
    def bates_option_price(self,K, params, S0, r, q, T):
        # 类似于 Heston 模型的定价，但使用 Bates 的特征函数
        def integrand(u):
            cf = model.bates_characteristic_function(u - 1j * 0.5, params, S0, r, q, T)
            numerator = np.exp(-1j * u * np.log(K)) * cf
            return numerator / (u ** 2 + 0.25)
        integral = quad(lambda u: integrand(u).real, 0, np.inf, limit=100)[0]
        price = S0 * np.exp(-q * T) - (np.sqrt(S0 * K) / np.pi) * np.exp(-r * T) * integral
        return price

    # 目标函数
    def bates_calibration_error(self,params):
        kappa, theta, sigma, rho, v0, lambd, muJ, sigmaJ = params
        model_prices = [self.bates_option_price(K, params, S0, r, q, T) for K in market_strikes]
        error = np.sum((model_prices - market_prices) ** 2)
        return error

# 市场数据（假设）
market_strikes = np.array([80, 90, 100, 110, 120])
market_prices = np.array([22, 14, 8, 5, 3])
S0 = 100
r = 0.05
q = 0.02
T = 1

# 初始猜测和参数边界
initial_guess = [1.0, 0.05, 0.5, -0.5, 0.05, 0.1, -0.1, 0.2]
bounds = [(0.0001, 10), (0.0001, 1), (0.0001, 5), (-0.999, 0.999), (0.0001, 1), (0.0001, 1), (-1, 1), (0.0001, 1)]

model = bates_model()
# 优化
result = minimize(model.bates_calibration_error, initial_guess, bounds=bounds, method='L-BFGS-B')

# 校准结果
params_calibrated = result.x
print(f"Calibrated parameters:\nKappa: {params_calibrated[0]}\nTheta: {params_calibrated[1]}\nSigma: {params_calibrated[2]}\nRho: {params_calibrated[3]}\nv0: {params_calibrated[4]}\nLambda: {params_calibrated[5]}\nMuJ: {params_calibrated[6]}\nSigmaJ: {params_calibrated[7]}")
```

### hetson model

```py
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad

# Heston 模型特征函数
def heston_characteristic_function(u, params, S0, r, q, T):
    kappa, theta, sigma, rho, v0 = params
    xi = kappa - rho * sigma * 1j * u
    d = np.sqrt(xi ** 2 + (sigma ** 2) * (u ** 2 + 1j * u))
    g = (xi - d) / (xi + d)
    C = r * 1j * u * T + (kappa * theta / sigma ** 2) * ((xi - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
    D = ((xi - d) / sigma ** 2) * (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
    return np.exp(C + D * v0 + 1j * u * np.log(S0))

# 期权价格积分
def heston_option_price(K, params, S0, r, q, T):
    def integrand(u):
        cf = heston_characteristic_function(u - 1j * 0.5, params, S0, r, q, T)
        numerator = np.exp(-1j * u * np.log(K)) * cf
        return numerator / (u ** 2 + 0.25)
    integral = quad(lambda u: integrand(u).real, 0, np.inf, limit=100)[0]
    price = S0 * np.exp(-q * T) - (np.sqrt(S0 * K) / np.pi) * np.exp(-r * T) * integral
    return price

# 市场数据（假设）
market_strikes = np.array([80, 90, 100, 110, 120])
market_prices = np.array([22, 14, 8, 5, 3])
S0 = 100  # 当前价格
r = 0.05  # 无风险利率
q = 0.02  # 股息率
T = 1     # 到期时间

# 目标函数
def heston_calibration_error(params):
    kappa, theta, sigma, rho, v0 = params
    model_prices = [heston_option_price(K, params, S0, r, q, T) for K in market_strikes]
    error = np.sum((model_prices - market_prices) ** 2)
    return error

# 初始猜测和参数边界
initial_guess = [1.0, 0.05, 0.5, -0.5, 0.05]
bounds = [(0.0001, 10), (0.0001, 1), (0.0001, 5), (-0.999, 0.999), (0.0001, 1)]

# 优化
result = minimize(heston_calibration_error, initial_guess, bounds=bounds, method='L-BFGS-B')

# 校准结果
kappa_cal, theta_cal, sigma_cal, rho_cal, v0_cal = result.x
print(f"Calibrated parameters:\nKappa: {kappa_cal}\nTheta: {theta_cal}\nSigma: {sigma_cal}\nRho: {rho_cal}\nv0: {v0_cal}")
```

### sabr model

```py
import numpy as np
from scipy.optimize import minimize
from scipy.special import erf

def sabr_implied_vol(F, K, T, alpha, beta, rho, nu):
    # SABR 模型的近似隐含波动率公式（Hagan 等人提出）
    epsilon = 1e-07
    if abs(F - K) < epsilon:
        logFK = epsilon
    else:
        logFK = np.log(F / K)

    F_mid = (F * K) ** ((1 - beta) / 2)
    z = (nu / alpha) * F_mid * logFK
    x_z = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))
    denom = F_mid * (1 + ((1 - beta) ** 2 / 24) * logFK ** 2 + ((1 - beta) ** 4 / 1920) * logFK ** 4)
    implied_vol = (alpha / denom) * (z / x_z) * (1 + (( ( ( (1 - beta) ** 2 ) / 24 ) * (alpha ** 2) / (F_mid ** (2 * (1 - beta))) ) + ( ( rho * beta * nu * alpha ) / (4 * F_mid ** (1 - beta)) ) + ( ( (2 - 3 * rho ** 2 ) * nu ** 2 ) / 24 )) * T)
    return implied_vol

# 市场数据（假设）
market_strikes = np.array([80, 90, 100, 110, 120])
market_vols = np.array([0.25, 0.22, 0.20, 0.22, 0.25])
F = 100  # 远期价格
T = 1    # 到期时间

# 目标函数
def sabr_calibration_error(params):
    alpha, beta, rho, nu = params
    model_vols = [sabr_implied_vol(F, K, T, alpha, beta, rho, nu) for K in market_strikes]
    error = np.sum((model_vols - market_vols) ** 2)
    return error

# 初始猜测和参数边界
initial_guess = [0.2, 0.5, 0.0, 0.2]
bounds = [(0.0001, 1), (0, 1), (-0.999, 0.999), (0.0001, 1)]

# 优化
result = minimize(sabr_calibration_error, initial_guess, bounds=bounds, method='L-BFGS-B')

# 校准结果
alpha_calibrated, beta_calibrated, rho_calibrated, nu_calibrated = result.x
print(f"Calibrated parameters:\nAlpha: {alpha_calibrated}\nBeta: {beta_calibrated}\nRho: {rho_calibrated}\nNu: {nu_calibrated}")
```