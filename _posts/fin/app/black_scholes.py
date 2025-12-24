import numpy as np
from scipy.stats import norm

def black_sholces_d1_d2(s,K,r,rf,volatility,T):
    d1 = (np.log(s/K)+(r-rf+0.5*volatility**2)*T)/(volatility*np.sqrt(T))
    d2 = d1 - volatility*np.sqrt(T)
    return d1 , d2

def black_shocles_call(s,K,r,rf,T):
    d1 , d2 = black_sholces_d1_d2(s,K,r,rf,T)
    return s*norm.cdf(d1)*np.exp(-rf*T)- K*np.exp(-r*T)*norm.cdf(d2)

def black_shocles_put(s,K,r,rf,T):
    d1 , d2 = black_sholces_d1_d2(s,K,r,rf,T)
    return -s*norm.cdf(-d1)*np.exp(-rf*T)+ K*np.exp(-r*T)*norm.cdf(-d2)
