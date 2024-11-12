---
layout: post
title:  非线性方程求解 牛顿-拉夫森法
date:   2024-11-08 11:24:29 +0800
categories:
    - python
    - na
---

牛顿-拉夫森法使用函数的导数加速迭代求根，适合收敛性较好的问题。

```py
def newton_raphson(func, deriv, x0, tol=1e-5, max_iter=100):
    for i in range(max_iter):
        f_x0 = func(x0)
        f_deriv_x0 = deriv(x0)
        if abs(f_deriv_x0) < tol:
            print("Derivative is zero, Newton-Raphson method fails")
            return None
        x1 = x0 - f_x0 / f_deriv_x0
        if abs(x1 - x0) < tol:
            return x1
        x0 = x1
    return x0

def func(x):
    return x**3 - x - 2

def deriv(x):
    return 3 * x**2 - 1

root = newton_raphson(func, deriv, 1.5)
print("Root found by Newton-Raphson method:", root)
```

牛顿法的迭代公式为 xn+1=xn−f(xn)f′(xn)xn+1​=xn​−f′(xn​)f(xn​)​，它的收敛速度比二分法和割线法更快，但需要函数的导数信息。