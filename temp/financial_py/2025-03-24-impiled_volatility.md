---
layout: post
title:  impiled volatility
date:   2025-03-24 11:24:29 +0800
categories: 
    - financial
    - python
---

```py
import pandas as pd
from scipy.stats import norm
import numpy as np

implied_volatility = pd.DataFrame([
  {
    "序号": 1,
    "标的资产价格": 150,
    "行权价": 150,
    "到期时间(年)": 0.0833,
    "无风险利率": 0.035,
    "市场期权价格": 7.50,
    "隐含波动率": "25%"
  },
  {
    "序号": 2,
    "标的资产价格": 150,
    "行权价": 155,
    "到期时间(年)": 0.0833,
    "无风险利率": 0.035,
    "市场期权价格": 6.20,
    "隐含波动率": "23%"
  },
  {
    "序号": 3,
    "标的资产价格": 150,
    "行权价": 145,
    "到期时间(年)": 0.0833,
    "无风险利率": 0.035,
    "市场期权价格": 8.80,
    "隐含波动率": "27%"
  },
  {
    "序号": 4,
    "标的资产价格": 150,
    "行权价": 150,
    "到期时间(年)": 0.25,
    "无风险利率": 0.03,
    "市场期权价格": 10.50,
    "隐含波动率": "22%"
  },
  {
    "序号": 5,
    "标的资产价格": 150,
    "行权价": 155,
    "到期时间(年)": 0.25,
    "无风险利率": 0.03,
    "市场期权价格": 9.30,
    "隐含波动率": "21%"
  },
  {
    "序号": 6,
    "标的资产价格": 150,
    "行权价": 145,
    "到期时间(年)": 0.25,
    "无风险利率": 0.03,
    "市场期权价格": 11.80,
    "隐含波动率": "24%"
  },
  {
    "序号": 7,
    "标的资产价格": 150,
    "行权价": 150,
    "到期时间(年)": 0.5,
    "无风险利率": 0.025,
    "市场期权价格": 15.20,
    "隐含波动率": "20%"
  },
  {
    "序号": 8,
    "标的资产价格": 150,
    "行权价": 155,
    "到期时间(年)": 0.5,
    "无风险利率": 0.025,
    "市场期权价格": 13.80,
    "隐含波动率": "19%"
  },
  {
    "序号": 9,
    "标的资产价格": 150,
    "行权价": 145,
    "到期时间(年)": 0.5,
    "无风险利率": 0.025,
    "市场期权价格": 16.80,
    "隐含波动率": "21%"
  },
  {
    "序号": 10,
    "标的资产价格": 150,
    "行权价": 150,
    "到期时间(年)": 1,
    "无风险利率": 0.02,
    "市场期权价格": 22.50,
    "隐含波动率": "18%"
  },
  {
    "序号": 11,
    "标的资产价格": 150,
    "行权价": 155,
    "到期时间(年)": 1,
    "无风险利率": 0.02,
    "市场期权价格": 20.30,
    "隐含波动率": "17%"
  }
])

implied_volatility.drop(columns="序号",inplace=True)

implied_volatility.columns = ['S','K','T','r','C','imp_v']
implied_volatility['imp_v'] = [float(implied_volatility['imp_v'][_].replace('%',''))/100 for _ in range(len(implied_volatility))]
print(implied_volatility)


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