---
layout: post
title:  black scholes model 
date:   2025-03-23 11:24:29 +0800
categories: 
    - financial
    - python
---

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
```

