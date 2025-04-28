---
layout: post
title:  zeromq black shocles model
date:   2024-11-07 11:24:29 +0800
categories: 
    - python
    - socket
---

在合理条件下,防止核心代码外泄是 `zeromq` 一个核心的运用

`client`

```py
import zmq

def request_calculation(data):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")
    
    # 发送数据
    print(f"Sending data to server: {data}")
    socket.send_json(data)
    
    # 接收结果
    result = socket.recv_json()
    print(f"Received result from server: {result}")
    
    return result

if __name__ == "__main__":

    data = {"S": 100, "K": 100, "T": 2, "r": 0.03 ,"q": 0, "volatility": 0.2}
    request_calculation(data)
```

`server.py`

```py
import zmq
import numpy as np
from scipy.stats import norm

def calculate(data):
    S = data['S']
    K = data['K']
    T = data['T']
    r = data['r']
    q = data['q']
    volatility = data['volatility']

    return black_scholes().black_scholes_call(S,K,T,r,q,volatility)

class black_scholes : 
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

def main():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    print("Server is ready to receive requests...")
    
    while True:
        message = socket.recv_json()
        print(f"Received request: {message}")
        
        result = calculate(message)
        
        socket.send_json(result)
        print(f"Sent result: {result}")

if __name__ == "__main__":

    main()
```