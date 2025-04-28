---
layout: post
title : convertible bond
date : 2024-10-30 11:24:29 +0800
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

### 可转换债券（convertible bond）

可转换债券的定价包含债券部分和转换期权部分。一般可采用二叉树模型或蒙特卡洛模拟来评估转换期权的价值。

```py
# 可转换债券的债券部分定价
def convertible_bond_price(face_value, coupon_rate, rate, years):
    coupon = face_value * coupon_rate
    bond_price = sum([coupon / (1 + rate) ** t for t in range(1, years + 1)]) + face_value / (1 + rate) ** years
    return bond_price

# 示例
face_value = 1000
coupon_rate = 0.04
rate = 0.05
years = 10
price = convertible_bond_price(face_value, coupon_rate, rate, years)
print(f"可转换债券（债券部分）价格: {price:.2f}")
```