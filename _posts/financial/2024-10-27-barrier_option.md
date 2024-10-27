---
title : barrier option 
date : 2024-10-27 11:24:29 +0800
categories: 
    - financial
    - option
---

### **barrier option**（障碍期权）  
   - 定价：当标的资产达到预定价格时，期权可能生效或失效。定价使用布莱克-舒尔斯模型结合反映障碍的条件。

存在解析解和 `monte carlo`

### 解析解

```py
@dataclass
class barrier_option :

    def _cal(self,S, K, T, r, volatility, B):
        self.lambda_ = (r + 0.5 * volatility**2) / volatility**2
        self.x1 = np.log(S / K) / (volatility * np.sqrt(T)) + (1 + self.lambda_) * volatility * np.sqrt(T)
        self.x2 = np.log(S / B) / (volatility * np.sqrt(T)) + (1 + self.lambda_) * volatility * np.sqrt(T)
        self.y1 = np.log(B**2 / (S * K)) / (volatility * np.sqrt(T)) + (1 + self.lambda_) * volatility * np.sqrt(T)
        self.y2 = np.log(B / S) / (volatility * np.sqrt(T)) + (1 + self.lambda_) * volatility * np.sqrt(T)
        
    def up_and_out_call(self,S, K, T, r, volatility, B):
        self._cal(S, K, T, r, volatility, B)
        if S >= B:
            return 0.0
        return S * norm.cdf(self.x1) - K * np.exp(-r * T) * norm.cdf(self.x1 - volatility * np.sqrt(T)) - \
            S * (B / S)**(2 * self.lambda_) * norm.cdf(self.y1) + \
            K * np.exp(-r * T) * (B / S)**(2 * self.lambda_ - 2) * norm.cdf(self.y1 - volatility * np.sqrt(T))

    def up_and_out_put(self,S, K, T, r, volatility, B):
        self._cal(S, K, T, r, volatility, B)
        if S >= B:
            return 0.0
        return K * np.exp(-r * T) * norm.cdf(-self.x1 + volatility * np.sqrt(T)) - S * norm.cdf(-self.x1) - \
            K * np.exp(-r * T) * (B / S)**(2 * self.lambda_ - 2) * norm.cdf(-self.y1 + volatility * np.sqrt(T)) + \
            S * (B / S)**(2 * self.lambda_) * norm.cdf(-self.y1)

    def down_and_out_call(self,S, K, T, r, volatility, B): 
        self._cal(S, K, T, r, volatility, B)
        if S <= B:
            return 0.0
        return S * norm.cdf(self.x1) - K * np.exp(-r * T) * norm.cdf(self.x1 - volatility * np.sqrt(T)) - \
            S * (B / S)**(2 * self.lambda_) * norm.cdf(self.y1) + \
            K * np.exp(-r * T) * (B / S)**(2 * self.lambda_ - 2) * norm.cdf(self.y1 - volatility * np.sqrt(T))

    def down_and_out_put(self,S, K, T, r, volatility, B): 
        self._cal(S, K, T, r, volatility, B)
        if S <= B:
            return 0.0
        return K * np.exp(-r * T) * norm.cdf(-self.x1 + volatility * np.sqrt(T)) - S * norm.cdf(-self.x1) - \
            K * np.exp(-r * T) * (B / S)**(2 * self.lambda_ - 2) * norm.cdf(-self.y1 + volatility * np.sqrt(T)) + \
            S * (B / S)**(2 * self.lambda_) * norm.cdf(-self.y1)

    def down_and_in_call(self,S, K, T, r, volatility, B):
        self._cal(S, K, T, r, volatility, B)
        if S <= B:
            return S * norm.cdf(self.x1) - K * np.exp(-r * T) * norm.cdf(self.x1 - volatility * np.sqrt(T))
        return S * (B / S)**(2 * self.lambda_) * norm.cdf(self.y1) - \
            K * np.exp(-r * T) * (B / S)**(2 * self.lambda_ - 2) * norm.cdf(self.y1 - volatility * np.sqrt(T))

    def down_and_in_put(self,S, K, T, r, volatility, B):
        self._cal(S, K, T, r, volatility, B)
        if S <= B:
            return K * np.exp(-r * T) * norm.cdf(-self.x1 + volatility * np.sqrt(T)) - S * norm.cdf(-self.x1)
        return K * np.exp(-r * T) * (B / S)**(2 * self.lambda_ - 2) * norm.cdf(-self.y1 + volatility * np.sqrt(T)) - \
            S * (B / S)**(2 * self.lambda_) * norm.cdf(-self.y1)

    def up_and_in_call(self,S, K, T, r, volatility, B):
        self._cal(S, K, T, r, volatility, B)
        if S >= B:
            return S * norm.cdf(self.x1) - K * np.exp(-r * T) * norm.cdf(self.x1 - volatility * np.sqrt(T))
        return S * (B / S)**(2 * self.lambda_) * norm.cdf(self.y1) - \
            K * np.exp(-r * T) * (B / S)**(2 * self.lambda_ - 2) * norm.cdf(self.y1 - volatility * np.sqrt(T))
    
    def up_and_in_put(self,S, K, T, r, volatility, B):
        self._cal(S, K, T, r, volatility, B)
        if S >= B:
            return K * np.exp(-r * T) * norm.cdf(-self.x1 + volatility * np.sqrt(T)) - S * norm.cdf(-self.x1)
        return K * np.exp(-r * T) * (B / S)**(2 * self.lambda_ - 2) * norm.cdf(-self.y1 + volatility * np.sqrt(T)) - \
            S * (B / S)**(2 * self.lambda_) * norm.cdf(-self.y1)

bo = barrier_option()

print("Up-and-Out Call Option Price:", bo.up_and_out_call(S, K, T, r, volatility, B))
print("Up-and-Out Put Option Price:", bo.up_and_out_put(S, K, T, r, volatility, B))
print("Down-and-Out Call Option Price:", bo.down_and_out_call(S, K, T, r, volatility, B))
print("Down-and-Out Put Option Price:", bo.down_and_out_put(S, K, T, r, volatility, B))
print("Up-and-In Call Option Price:", bo.up_and_in_call(S, K, T, r, volatility, B))
print("Up-and-In Put Option Price:", bo.up_and_in_put(S, K, T, r, volatility, B))
print("Down-and-In Call Option Price:", bo.down_and_in_call(S, K, T, r, volatility, B))
print("Down-and-In Put Option Price:", bo.down_and_in_put(S, K, T, r, volatility, B))
```

```py
S = 100.0     # 初始资产价格
K = 100.0    # 执行价格
T = 1.0       # 到期时间（以年计）
r = 0.05    # 无风险利率
volatility = 0.2 # 波动率
B = 110.0     # 障碍价格
```

```py
Up-and-Out Call Option Price : -20.84206953857337
Up-and-Out Put Option Price : 4.134968526768741
Down-and-Out Call Option Price : 0.0
Down-and-Out Put Option Price : 0.0
Up-and-In Call Option Price : 31.14889547751882
Up-and-In Put Option Price : 1.2947998622481123
Down-and-In Call Option Price : 10.306825938945451
Down-and-In Put Option Price : 5.4297683890168535
```

<!-- ### monte carlo -->

<!-- ```py
import numpy as np

``` -->

### 日志

2024-10-27 : barrier option monte carlo 和解析解有差别,可能公式错误或者代码错误,需要修正代码