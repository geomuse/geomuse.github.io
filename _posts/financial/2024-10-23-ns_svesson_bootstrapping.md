---
layout: post
title:  ns model and svensson model
date:   2024-10-23 11:24:29 +0800
categories: 
    - financial
    - bootstrapping
---

<!-- 

nelson and siegel model 

svensson model 

-->

```py
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as pt
from matplotlib import style
style.use('ggplot')

def nelson_siegel(t, beta0, beta1, beta2, lambd):
    term1 = beta0
    term2 = beta1 * (1 - np.exp(-lambd * t)) / (lambd * t)
    term3 = beta2 * ((1 - np.exp(-lambd * t)) / (lambd * t) - np.exp(-lambd * t))
    return term1 + term2 + term3

def svensson(t, beta0, beta1, beta2, beta3, lambd1, lambd2):
    term1 = beta0
    term2 = beta1 * (1 - np.exp(-lambd1 * t)) / (lambd1 * t)
    term3 = beta2 * ((1 - np.exp(-lambd1 * t)) / (lambd1 * t) - np.exp(-lambd1 * t))
    term4 = beta3 * ((1 - np.exp(-lambd2 * t)) / (lambd2 * t) - np.exp(-lambd2 * t))
    return term1 + term2 + term3 + term4

# 生成模拟的市场数据
times = np.array([0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])  # 期限 (年)
true_rates = np.array([0.01, 0.015, 0.02, 0.022, 0.025, 0.027, 0.03, 0.032, 0.033, 0.034])  # 模拟的市场利率

# 初始猜测参数
initial_params_ns = [0.03, -0.02, 0.02, 1.0]  # beta0, beta1, beta2, lambd
params_ns, _ = curve_fit(nelson_siegel, times, true_rates, p0=initial_params_ns)

# 预测拟合结果
fitted_ns_rates = nelson_siegel(times, *params_ns)
print("Nelson-Siegel 模型拟合参数: ", params_ns)

# 初始猜测参数
initial_params_sv = [0.03, -0.02, 0.02, 0.01, 1.0, 3.0]  # beta0, beta1, beta2, beta3, lambd1, lambd2
params_sv, _ = curve_fit(svensson, times, true_rates, p0=initial_params_sv)
fitted_sv_rates = svensson(times, *params_sv)
print("Svensson 模型拟合参数: ", params_sv)

# 绘制拟合曲线
pt.plot(times, true_rates, 'o', label="市场数据")
pt.plot(times, fitted_ns_rates, label="Nelson-Siegel 拟合")
pt.plot(times, fitted_sv_rates, label="Svensson 拟合")
pt.xlabel("到期期限 (年)")
pt.ylabel("利率")
pt.title("Nelson-Siegel 和 Svensson 模型拟合")
pt.legend()
pt.show()
```