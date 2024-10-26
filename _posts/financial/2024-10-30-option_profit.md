---
layout: post
title : option profit graph
date : 2024-10-30 11:24:29 +0800
categories: 
    - financial
    - option
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

要建立通用的期权收益图代码，可以使用 Python 的 matplotlib 库进行绘图。下面的代码支持绘制看涨期权（call）、看跌期权（put）、向上敲出期权（up-and-out）、向下敲出期权（down-and-out）等不同类型的期权收益图。

```py
import numpy as np
import matplotlib.pyplot as pt

# 定义期权参数
K = 50  # 行权价
B = 60  # 敲出价格
C = 5   # 期权费用
S = np.linspace(0, 100, 500)  # 标的价格区间

# 计算各类期权的收益函数
def call_payoff(S, K, C):
    return np.maximum(S - K, 0) - C

def put_payoff(S, K, C):
    return np.maximum(K - S, 0) - C

def up_and_out_call(S, K, B, C):
    payoff = np.maximum(S - K, 0)
    payoff[S >= B] = 0  # 超过敲出价格收益为0
    return payoff - C

def down_and_out_put(S, K, B, C):
    payoff = np.maximum(K - S, 0)
    payoff[S <= B] = 0  # 低于敲出价格收益为0
    return payoff - C

# 绘制图像
pt.figure(figsize=(10, 6))

# 看涨期权
pt.plot(S, call_payoff(S, K, C), label="Call Option", color="blue")
# 看跌期权
pt.plot(S, put_payoff(S, K, C), label="Put Option", color="red")
# 向上敲出看涨期权
pt.plot(S, up_and_out_call(S, K, B, C), label="Up-and-Out Call", color="green")
# 向下敲出看跌期权
pt.plot(S, down_and_out_put(S, K, B, C), label="Down-and-Out Put", color="purple")

# 设置图像标签和标题
pt.xlabel("Stock Price at Expiration")
pt.ylabel("Payoff")
pt.title("Payoff Diagram for Various Options")
pt.legend()
pt.grid(True)
pt.show()
```

`call_payoff` : 计算看涨期权的收益。
`put_payoff` : 计算看跌期权的收益。
`up_and_out_call` : 计算向上敲出看涨期权的收益，若价格超过敲出价则收益为零。
`down_and_out_put` : 计算向下敲出看跌期权的收益，若价格低于敲出价则收益为零。