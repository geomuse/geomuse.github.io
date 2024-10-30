---
layout: post
title : puttable bond
date : 2024-11-02 11:24:29 +0800
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

### 可回售债券（Puttable Bond）

可回售债券允许投资者在市场条件不佳时要求发行人回购债券。类似于可赎回债券的定价，这里也需要考虑期权价值。

```py
def puttable_bond_price(face_value, coupon_rate, rate, years):
    coupon = face_value * coupon_rate
    bond_price = sum([coupon / (1 + rate) ** t for t in range(1, years + 1)]) + face_value / (1 + rate) ** years
    # 需加上回售期权价值，此处略去
    return bond_price

# 示例
face_value = 1000
coupon_rate = 0.04
rate = 0.05
years = 10
price = puttable_bond_price(face_value, coupon_rate, rate, years)
print(f"可回售债券（债券部分）价格: {price:.2f}")
```