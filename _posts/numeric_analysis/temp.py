import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as pt

# 假设有一组测量数据 y 和预测值 y_pred
y = np.array([10.2, 9.9, 10.5, 10.3, 9.8])   # 实际测量值
y_pred = np.array([10.0, 10.1, 10.3, 10.4, 9.9])  # 模型预测值

# 残差分析
residuals = y - y_pred
mean_residual = np.mean(residuals)
std_residual = np.std(residuals)

# 正态性检验
_ , p_value = stats.shapiro(residuals)  # p值 < 0.05 则拒绝正态性假设

# 计算误差传播 - 假设 y_pred 的不确定度为 0.1
uncertainty_y_pred = 0.1
combined_uncertainty = np.sqrt(uncertainty_y_pred**2 + std_residual**2)

# 输出结果
print(f"残差均值(Mean Residual): {mean_residual}")
print(f"残差标准差(Residual Std Dev): {std_residual}")
print(f"正态性检验 p 值: {p_value}")
print(f"预测结果的综合不确定度: {combined_uncertainty}")

# 残差分布图
pt.hist(residuals, bins=5, alpha=0.6, color='blue')
pt.title('Residual Distribution')
pt.xlabel('Residual')
pt.ylabel('Frequency')
pt.show()
