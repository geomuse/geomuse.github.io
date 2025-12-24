import numpy as np 

def binomial_tree_stock_price(so,K,T,r,volatility,n):
    dT = T/n
    u = np.exp(volatility*np.sqrt(dT))
    d = 1/u
    p  = (np.exp(r*dT) - d) / (u -d)

    sT = np.array([so * (u ** (n - i)) * (d ** i) for i in range(n + 1)])
    return sT , p , dT

def binomial_tree_call(so,K,T,r,volatility,n):
    sT , p , dT = binomial_tree_stock_price(so,K,T,r,volatility,n)
    option = np.maximum(sT - K, 0)
    for _ in range(n):
        option = np.exp(-r * dT) * (p * option[:-1] + (1 - p) * option[1:])
    return  option[0]

def binomial_tree_put(so,K,T,r,volatility,n):
    sT , p , dT = binomial_tree_stock_price(so,K,T,r,volatility,n)
    option = np.maximum(K-sT, 0)
    for _ in range(n):
        option = np.exp(-r * dT) * (p * option[:-1] + (1 - p) * option[1:])
    return  option[0]

if __name__ == '__main__' : 

    so = 100 
    K = 100
    T = 1
    r = 0.05
    rf = 0.0
    volatility = 10/100

    print(binomial_tree_call(so,K,T,r,volatility,300))

    option = []
    for _ in range(10,500,10):
        option.append(binomial_tree_call(so,K,T,r,volatility,_))

    import matplotlib.pyplot as pt

    pt.plot(np.array(option))
    pt.show()

