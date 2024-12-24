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

S = 100.0     # 初始资产价格
K = 100.0    # 执行价格
T = 1.0       # 到期时间（以年计）
r = 0.05    # 无风险利率
volatility = 0.2 # 波动率
B = 110.0     # 障碍价格

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
Up-and-Out Call Option Price : -20.84206953857337
Up-and-Out Put Option Price : 4.134968526768741
Down-and-Out Call Option Price : 0.0
Down-and-Out Put Option Price : 0.0
Up-and-In Call Option Price : 31.14889547751882
Up-and-In Put Option Price : 1.2947998622481123
Down-and-In Call Option Price : 10.306825938945451
Down-and-In Put Option Price : 5.4297683890168535
```

### monte carlo

```py
import numpy as np

def monte_carlo_barrier_option(S, K, T, r, volatility, B, option_type, barrier_type, num_paths=10000, num_steps=100):

    dt = T / num_steps  # 每一步的时间间隔
    discount_factor = np.exp(-r * T)  # 折现因子
    payoff_sum = 0.0  # 总的期权收益

    for _ in range(num_paths):
        # 初始化路径
        path = [S]
        barrier_breached = False

        # 生成路径
        for _ in range(num_steps):
            z = np.random.standard_normal()
            S_t = path[-1] * np.exp((r - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * z)
            path.append(S_t)

            # 检查障碍触发条件
            if barrier_type == 'up_and_out' and S_t >= B:
                barrier_breached = True
                break
            elif barrier_type == 'down_and_out' and S_t <= B:
                barrier_breached = True
                break
            elif barrier_type == 'up_and_in' and S_t >= B:
                barrier_breached = True
            elif barrier_type == 'down_and_in' and S_t <= B:
                barrier_breached = True

        # 计算期权的收益
        if barrier_type in ['up_and_out', 'down_and_out']:
            # 出局类型，若障碍触发收益为0
            if not barrier_breached:
                if option_type == 'call':
                    payoff = max(path[-1] - K, 0)  # 看涨期权
                elif option_type == 'put':
                    payoff = max(K - path[-1], 0)  # 看跌期权
                payoff_sum += payoff
        elif barrier_type in ['up_and_in', 'down_and_in']:
            # 入局类型，若障碍未触发收益为0
            if barrier_breached:
                if option_type == 'call':
                    payoff = max(path[-1] - K, 0)  # 看涨期权
                elif option_type == 'put':
                    payoff = max(K - path[-1], 0)  # 看跌期权
                payoff_sum += payoff

    # 计算期权价格（期望收益的折现值）
    price = (payoff_sum / num_paths) * discount_factor
    return price

# 示例参数
S = 100.0     # 初始资产价格
K = 100.0     # 执行价格
T = 1.0       # 到期时间（以年计）
r = 0.05    # 无风险利率
volatility = 0.2 # 波动率
B = 110.0     # 障碍价格

print("Up-and-Out Call Option Price:", monte_carlo_barrier_option(S, K, T, r, volatility, B, 'call', 'up_and_out'))
print("Up-and-Out Put Option Price:", monte_carlo_barrier_option(S, K, T, r, volatility, B, 'put', 'up_and_out'))
print("Down-and-Out Call Option Price:", monte_carlo_barrier_option(S, K, T, r, volatility, B, 'call', 'down_and_out'))
print("Down-and-Out Put Option Price:", monte_carlo_barrier_option(S, K, T, r, volatility, B, 'put', 'down_and_out'))
print("Up-and-In Call Option Price:", monte_carlo_barrier_option(S, K, T, r, volatility, B, 'call', 'up_and_in'))
print("Up-and-In Put Option Price:", monte_carlo_barrier_option(S, K, T, r, volatility, B, 'put', 'up_and_in'))
print("Down-and-In Call Option Price:", monte_carlo_barrier_option(S, K, T, r, volatility, B, 'call', 'down_and_in'))
print("Down-and-In Put Option Price:", monte_carlo_barrier_option(S, K, T, r, volatility, B, 'put', 'down_and_in'))
``` 

### 日志

2024-10-27 : barrier option monte carlo 和解析解有差别,可能公式错误或者代码错误,需要修正代码