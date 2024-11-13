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