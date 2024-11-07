---
layout: post
title:  非线性方程求解 二分法和割线法
date:   2024-11-07 11:24:29 +0800
categories:
    - python
    - na
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

1. 二分法

二分法用于求解方程 $f(x)=0$ 的根。它需要一个区间 $[a,b]$，并且在该区间上 f(a)⋅f(b)<0f(a)⋅f(b)<0。

```py
def bisection_method(func, a, b, tol=1e-5, max_iter=100):
    if func(a) * func(b) >= 0:
        print("Bisection method fails.")
        return None
    for i in range(max_iter):
        c = (a + b) / 2
        if abs(func(c)) < tol:
            return c
        elif func(a) * func(c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2

# 示例
def func(x):
    return x**3 - x - 2

root = bisection_method(func, 1, 2)
print("Root found by bisection method:", root)
```

解释：
二分法在每次迭代中将区间长度减半，直到找到足够精确的根。每次迭代检查中点 c 是否满足 f(c)=0f(c)=0 或达到容差。
2. 割线法

割线法是一种不需要求导数的迭代方法，比二分法收敛速度快，但要求初始两个点。

```py
def secant_method(func, x0, x1, tol=1e-5, max_iter=100):
    for i in range(max_iter):
        f_x0 = func(x0)
        f_x1 = func(x1)
        if abs(f_x1 - f_x0) < tol:
            print("Divide by zero in secant method")
            return None
        x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        if abs(x2 - x1) < tol:
            return x2
        x0, x1 = x1, x2
    return x1

# 示例
root = secant_method(func, 1, 2)
print("Root found by secant method:", root)
```

解释：
割线法利用两个点 x0 和 x1 生成割线，并逐步逼近根。相比于二分法，割线法没有区间要求，但可能收敛性不如牛顿法。