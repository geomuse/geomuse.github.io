---
layout: post
title:  volatility 
date:   2024-10-25 11:24:29 +0800
categories: 
    - financial
    - volatility
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

call 与 put 在相同的执行价格 K , T 其波动率一样

波动率微笑定义为隐含波动率与 `K/So` `K/Fo` , $\Delta$ 之间关系

问题 : 每个点都是一份合约 ?

有了市场上的波动率就通过插值得到非标准期限的波动率

或者通过市场拟合后的 `heston model` , `bates model` , `sabr model` 方法

以下是市场的波动率的3D图形

```py
import numpy as np
import matplotlib.pyplot as pt
from mpl_toolkits.mplot3d import Axes3D

# 您提供的数据
vol = np.array([[14.2,13.0,12.0,13.1,14.5],
                [14.0,13.0,12.0,13.1,14.2],
                [14.1,13.3,12.5,13.4,14.3],
                [14.7,14.0,13.5,14.0,14.8],
                [15.0,14.4,14.0,14.5,15.1],
                [14.8,14.6,14.4,14.7,15.0]
                ])

T= np.array([1/12,3/12,0.5,1,2,5])

# 假设的行权价数组（根据数据的列数）
K = np.array([80, 90, 100, 110, 120]) / 100

# 创建网格
K_grid, T_grid = np.meshgrid(K, T)

# 绘制3D图形
fig = pt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')

# 将波动率转换为浮点数并除以100以表示为小数
vol_float = vol.astype(np.float64)

surf = ax.plot_surface(K_grid, T_grid, vol_float, cmap='viridis')

ax.set_xlabel('K/So', labelpad=15)
ax.set_ylabel('T(years)', labelpad=15)
ax.set_zlabel('implied volatility(%)', labelpad=15)
ax.set_title('volatility smile', pad=20)

fig.colorbar(surf, shrink=0.5, aspect=5)

pt.show()
```