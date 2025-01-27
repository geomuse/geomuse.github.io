---
layout: post
title : callable bond
date : 2024-10-31 11:24:29 +0800
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

### 可赎回债券（Callable Bond）

可赎回债券的定价包含债券价值和赎回期权的估值。为了简化，这里只给出债券现值的计算，实际计算通常需要期权定价模型。

```py
def callable_bond_price(face_value, coupon_rate, rate, years):
    coupon = face_value * coupon_rate
    bond_price = sum([coupon / (1 + rate) ** t for t in range(1, years + 1)]) + face_value / (1 + rate) ** years
    # 需减去赎回期权价值，此处略去
    return bond_price

# 示例
face_value = 1000
coupon_rate = 0.04
rate = 0.05
years = 10
price = callable_bond_price(face_value, coupon_rate, rate, years)
print(f"可赎回债券（债券部分）价格: {price:.2f}")
```