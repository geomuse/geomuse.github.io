---
layout: post
title:  black scholes
date:   2025-04-29 11:24:29 +0800
categories: 
    - review 
    - financial
    - python
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

## Black-Scholes 模型简介

**Black-Scholes模型**（又称Black-Scholes-Merton模型）是1973年由Fischer Black、Myron Scholes提出，后来Robert Merton完善的，用来定价欧式期权（只能在到期日行权）的一种经典金融数学模型。

Black-Scholes公式推导出了在某些假设条件下，期权价格应如何随时间和标的资产价格变化。  
它也奠定了现代金融衍生品定价的基础。

---

## Black-Scholes模型的**前提假设**

1. **标的资产价格服从几何布朗运动**  
   - 即资产价格的对数收益率是正态分布的。
   - 资产价格 $ S_t $ 的变化可用随机微分方程描述：
     $$
     dS_t = \mu S_t dt + \sigma S_t dW_t
     $$
     其中 $ \mu $ 是期望收益率，$ \sigma $ 是波动率，$ dW_t $ 是标准布朗运动。

2. **无套利市场**  
   - 市场不存在无风险套利机会。

3. **连续交易**  
   - 投资者可以随时进行买卖，且无任何交易限制。

4. **无交易成本和税收**  
   - 买卖资产和期权都没有手续费或税金。

5. **可无风险地借贷和存款**  
   - 投资者可以以固定无风险利率 $ r $ 随意借入或贷出资金。

6. **资产不支付红利**  
   - 原版模型假设标的资产（如股票）期间没有分红。

7. **利率恒定且已知**  
   - 无风险利率 $ r $ 是已知并且在整个期权存续期间保持不变。

8. **波动率恒定且已知**  
   - 标的资产的波动率 $ \sigma $ 是已知且不随时间变化。

---

## Black-Scholes模型的**局限性**

尽管非常成功，Black-Scholes模型在实际应用中存在一些重要的**局限性**：

1. **波动率并非恒定**
   - 实际市场上，波动率是变化的，并且会随着时间和资产价格水平改变（即“波动率微笑”现象）。

2. **资产价格跳跃**
   - 现实中，资产价格可能因重大新闻或事件而出现跳跃，而不是连续变化（违反了布朗运动连续性假设）。

3. **交易成本和税收**
   - 真实交易中存在买卖成本和税负，而模型忽略了这一点。

4. **无法精确对冲**
   - 连续调整持仓以对冲是理论假设，实际中只能离散地调整，因此存在“对冲误差”。

5. **无风险利率和波动率的变化**
   - 利率和波动率在现实中并不是固定不变的。

6. **不能处理美式期权**
   - 原版Black-Scholes只适用于**欧式期权**（到期才能行权），不适用于可以提前行权的**美式期权**。

7. **不适用于有分红股票（除非调整）**
   - 如果标的资产支付分红，需对模型进行修正（比如引入分红率）。

8. **极端事件（黑天鹅事件）**
   - Black-Scholes模型无法很好地捕捉极端市场事件，像2008年金融危机那样的暴跌情况。

```py
from scipy.stats import norm
import numpy as np

class black_scholes :

   def __d1_d2(self,S,K,T,r,sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1 , d2

   def call(self,S, K, T, r, sigma):
        d1 , d2 = self.__d1_d2(S,K,T,r,sigma)
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price

   def put(self,S, K, T, r, sigma):
        d1 , d2 = self.__d1_d2(S,K,T,r,sigma)
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return put_price
    
   def call_greeks(self,S, K, T, r, sigma):
        d1 , d2 = self.__d1_d2(S,K,T,r,sigma)
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * norm.pdf(d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                - r * K * np.exp(-r * T) * norm.cdf(d2))
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        return delta, gamma, vega, theta, rho

   def put_greeks(self,S, K, T, r, sigma):
        d1 , d2 = self.__d1_d2(S,K,T,r,sigma)
        delta = norm.cdf(d1) - 1
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * norm.pdf(d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                + r * K * np.exp(-r * T) * norm.cdf(-d2))
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        return delta, gamma, vega, theta, rho

   def impiled_volaility(self,S,K,T,r,c):
      f = lambda sigma : c - self.call(S,K,T,r,sigma)
      return f

def bisection(a,b,tol,no,f):
      fa = f(a)
      i = 0
      while i <= no :
          p = a + (b-a)/2
          fp = f(p)
          yield p
          if fp == 0 or (b-a)/2 < tol :
              return  
          i+=1 
          if fa*fp > 0 :
              a=p 
              fa=fp
          else :
              b=p
```

```py
if __name__ == '__main__':
    
  f = black_scholes().impiled_volaility(150,150,0.0833,0.035,7.5)
  volatility = 0.4223609470500378
  print(f(volatility))
  for r in bisection(-5,5,1e-10,500,f):
      print(r) 
  print(black_scholes().call(150,150,0.0833,0.035,volatility))
```

「实际应用中常见的Black-Scholes改进模型」，比如：
- **局部波动率模型（如Dupire方程）**
- **随机波动率模型（如Heston模型）**
- **跳跃扩散模型（如Merton Jump-Diffusion Model）**
