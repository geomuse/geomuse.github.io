---
layout: post
title:  option pricing 
date:   2024-10-23 11:24:29 +0800
categories: 
    - financial
    - option
---

```py
import numpy as np
from scipy.stats import norm

class Black_scholes : 
    def __d1(self,S,K,T,r,q,sigma):
        return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    def __d2(self,S,K,T,r,q,sigma):
        return self.__d1(S,K,T,r,q,sigma) - sigma * np.sqrt(T)

    def black_scholes_call(self,S,K,T,r,q,sigma):
        d1 , d2 = self.__d1(S,K,T,r,q,sigma) , self.__d2(S,K,T,r,q,sigma)
        return S * np.exp(-q*T) * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

    def black_scholes_put(self,S,K,T,r,q,sigma):
        d1 , d2 = self.__d1(S,K,T,r,q,sigma) , self.__d2(S,K,T,r,q,sigma)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
```
