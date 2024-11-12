---
layout: post
title : floating rate bond 
date : 2024-10-29 11:24:29 +0800
categories: 
    - financial
    - bond
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

### 浮动利率债券 (floating rate bond)

浮动利率债券的息票支付会随市场利率波动，因此要考虑每个重置期的基准利率。通常情况下可以假设未来基准利率不变，但实际应用中往往需要模拟利率变化。

$$\pi = \sum_{i=1}^{n}\frac{c}{(1+r)^t} + \frac{m}{(1+r)^t}$$

```py
def floating_rate_bond_price(face_value, margin, base_rate_list, years):
    total_price = 0
    for t in range(1, years + 1):
        coupon = face_value * (base_rate_list[t - 1] + margin)
        total_price += coupon / (1 + base_rate_list[t - 1] + margin) ** t
    return total_price

face_value = 1000
margin = 0.02
base_rate_list = [0.03, 0.035, 0.04, 0.037, 0.032]  # 每年的基准利率
years = 5
price = floating_rate_bond_price(face_value, margin, base_rate_list, years)
print(f"浮动利率债券价格: {price:.2f}")
```