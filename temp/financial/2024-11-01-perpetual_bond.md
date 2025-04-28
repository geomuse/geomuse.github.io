---
layout: post
title : perpetual bond
date : 2024-11-01 11:24:29 +0800
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

### 永续债券（Perpetual Bond）

永续债券没有到期日，因此只需计算其无限期现金流的现值即可。

```py
def perpetual_bond_price(coupon, rate):
    return coupon / rate

# 示例
coupon = 50
rate = 0.05
price = perpetual_bond_price(coupon, rate)
print(f"永续债券价格: {price:.2f}")
```