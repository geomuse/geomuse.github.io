---
layout: post
title : leveraged loan
date : 2024-11-03 11:24:29 +0800
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

### 杠杆贷款债券（Leveraged Loan）

杠杆贷款债券通常考虑高信用风险，基本定价方法与普通息票债券类似，但需要加入较大的信用利差。

```py
def leveraged_loan_price(face_value, coupon_rate, rate, credit_spread, years):
    rate += credit_spread  # 加上信用利差
    coupon = face_value * coupon_rate
    price = sum([coupon / (1 + rate) ** t for t in range(1, years + 1)]) + face_value / (1 + rate) ** years
    return price

# 示例
face_value = 1000
coupon_rate = 0.04
rate = 0.05
credit_spread = 0.03
years = 10
price = leveraged_loan_price(face_value, coupon_rate, rate, credit_spread, years)
print(f"杠杆贷款债券价格: {price:.2f}")
```