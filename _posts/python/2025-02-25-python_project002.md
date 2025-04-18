---
layout: post
title: python 奇偶数判断 专案
date: 2025-02-25 10:24:29 +0800
categories:
    - python
---

任务：

编写一个程序，要求用户输入一个整数，并判断它是 **奇数** 还是 **偶数**。

示例输入： 

```
请输入一个整数：7
```

示例输出：

```
7 是奇数
```

```py
import math 

def match_even_or_odd(num):
    match num :
        case _ if num % 2 == 0 : return "even"
        case _ : return "odd"

print(match_even_or_odd(7))
```