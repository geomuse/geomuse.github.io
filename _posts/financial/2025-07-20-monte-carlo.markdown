---
layout: post
title:  Monte Carlo
date:   2025-07-20 11:01:35 +0300
image:  02.jpg
tags:   Financial Option MONTECARLO
---

Monte Carlo 模拟是一种通过随机抽样估算期权定价期望的方法。其主要缺点是方差大、收敛慢(误差 ∼$\frac{1}{\sqrt{N}}$),因此发展了许多“方差缩减技术”来改善

| 方法名                    | 优点          | 用途        
| ---------------------- | ----------- | ---------- |
| 1. Basic MC             | 简单直观        | 欧式期权       
| 2. Antithetic Variates | 方差减半        | 欧式、障碍期权    
| 3. Control Variates    | 使用解析解减少误差   | 欧式、组合期权    
| 4. Moment Matching     | 保持均值方差一致    | 提高模拟稳定性    
| 5. Importance Sampling | 更快收敛        | 深度实值/虚值期权  
| 6. Quasi-MC            | 使用 Sobol 序列 | 提高高维收敛     
| 7. Longstaff-Schwartz  | 美式期权估价      | 美式、可提前赎回期权 
| 8. Barrier MC          | 模拟障碍触发过程    | 上/下敲出、敲入期权 

### Basic MC   

```py
import numpy as np

def basic_mc_call(S0, K, T, r, sigma, N=100000):
    Z = np.random.standard_normal(N)
    ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    payoff = np.maximum(ST - K, 0)
    return np.exp(-r*T) * np.mean(payoff)

print("Basic:", basic_mc_call(100, 100, 1, 0.05, 0.2))
```

### Antithetic Variates

```py
def antithetic_mc_call(S0, K, T, r, sigma, N=100000):
    Z = np.random.standard_normal(N//2)
    Z_all = np.concatenate([Z, -Z])
    ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z_all)
    payoff = np.maximum(ST - K, 0)
    return np.exp(-r*T) * np.mean(payoff)
```