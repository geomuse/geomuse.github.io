---
layout: post
title:  black model
date:   2025-07-22 11:01:35 +0300
image:  04.jpg
tags:   Financial Option Python
---

### Black 模型（Black 1976 或 Black ’76）

用于 **期货和远期合约上的欧式期权**，常用于利率衍生品（如 cap/floor、swaption）或商品期权。

**远期合约（forward）或期货合约（futures）**

#### 假设条件：

与 Black-Scholes 类似，但以远期价格作为基础资产。

#### Black 模型的看涨期权定价公式（以期货为标的）：

$$
C = e^{-rT} [F_0 N(d_1) - K N(d_2)]
$$

$$
d_1 = \frac{\ln(F_0 / K) + \frac{1}{2} \sigma^2 T}{\sigma \sqrt{T}}, \quad
d_2 = d_1 - \sigma \sqrt{T}
$$

其中：

* $F_0$：当前远期或期货价格
* $K$：执行价格
* 其他符号含义同上