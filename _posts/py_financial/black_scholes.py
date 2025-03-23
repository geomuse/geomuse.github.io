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

if __name__ == '__main__':

    # 示例参数
    S = 100      # 当前股票价格
    S = np.arange(50,150,1)
    K = 100      # 执行价格
    T = 1        # 到期时间（1年）
    r = 0.05     # 无风险利率5%
    sigma = 0.2  # 波动率20%
    
    # call_price = black_scholes().call(S, K, T, r, sigma)
    # put_price = black_scholes().put(S, K, T, r, sigma)

    # print(f"欧式看涨期权价格: {call_price:.2f}")
    # print(f"欧式看跌期权价格: {put_price:.2f}")
    delta, gamma, vega, theta, rho = black_scholes().call_greeks(S, K, T, r, sigma)
     
    import matplotlib.pyplot as pt
    pt.style.use("ggplot")

    pt.subplot(5, 1, 1)
    pt.plot(delta)
    pt.title("delta")

    pt.subplot(5, 1, 2)
    pt.plot(gamma)
    pt.title("gamma")

    pt.subplot(5, 1, 3)
    pt.plot(vega)
    pt.title("vega")

    pt.subplot(5, 1, 4)
    pt.plot(theta)
    pt.title("theta")

    pt.subplot(5, 1, 5)
    pt.plot(rho)
    pt.title("rho")

    pt.show()