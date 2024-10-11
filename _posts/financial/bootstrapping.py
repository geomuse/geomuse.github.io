import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
from scipy.optimize import minimize

# 定义Nelson-Siegel-Svensson模型函数
def svensson(tau, beta0, beta1, beta2, beta3, lambda1, lambda2):
    term1 = beta0
    term2 = beta1 * ((1 - np.exp(-tau / lambda1)) / (tau / lambda1))
    term3 = beta2 * (((1 - np.exp(-tau / lambda1)) / (tau / lambda1)) - np.exp(-tau / lambda1))
    term4 = beta3 * (((1 - np.exp(-tau / lambda2)) / (tau / lambda2)) - np.exp(-tau / lambda2))
    return term1 + term2 + term3 + term4

# 定义目标函数
def objective_function(params, tau, y):
    beta0, beta1, beta2, beta3, lambda1, lambda2 = params
    y_model = svensson(tau, beta0, beta1, beta2, beta3, lambda1, lambda2)
    error = y - y_model
    return np.sum(error ** 2)

# 市场数据（期限：年，收益率：%）
data = pd.DataFrame({
    'tau': [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30],
    'yield': [1.5, 1.7, 1.9, 2.1, 2.3, 2.7, 3.0, 3.2, 3.5, 3.7]
})

tau = data['tau'].values
y = data['yield'].values

# 初始参数猜测
initial_params = [3.0, -1.0, 1.0, 0.5, 1.0, 3.0]

# 参数约束和边界
bounds = (
    (None, None),  # beta0
    (None, None),  # beta1
    (None, None),  # beta2
    (None, None),  # beta3
    (0.0001, None),  # lambda1
    (0.0001, None)   # lambda2
)

# 优化参数
result = minimize(objective_function, initial_params, args=(tau, y), bounds=bounds, method='L-BFGS-B')

beta0_opt, beta1_opt, beta2_opt, beta3_opt, lambda1_opt, lambda2_opt = result.x

print("优化后的参数：")
print(f"beta0 = {beta0_opt}")
print(f"beta1 = {beta1_opt}")
print(f"beta2 = {beta2_opt}")
print(f"beta3 = {beta3_opt}")
print(f"lambda1 = {lambda1_opt}")
print(f"lambda2 = {lambda2_opt}")

# 计算拟合后的收益率
tau_fit = np.linspace(0.1, 30, 300)
y_fit = svensson(tau_fit, beta0_opt, beta1_opt, beta2_opt, beta3_opt, lambda1_opt, lambda2_opt)

# 绘制结果
pt.figure(figsize=(10, 6))
pt.plot(tau, y, 'o', label='市场收益率')
pt.plot(tau_fit, y_fit, '-', label='NS Svensson模型拟合')
pt.xlabel('期限（年）')
pt.ylabel('收益率（%）')
pt.title('Nelson-Siegel-Svensson模型拟合利率曲线')
pt.legend()
pt.grid(True)
pt.show()