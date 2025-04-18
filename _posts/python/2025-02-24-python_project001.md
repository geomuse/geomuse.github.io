---
layout: post
title: python BMI 专案
date: 2025-02-24 10:24:29 +0800
categories:
    - python
---


任务：

编写一个 Python 程序，输入 身高（m） 和 体重（kg），计算 BMI 指数（体重 ÷ 身高²）。

根据 BMI 值，输出健康状态：

```
BMI < 18.5: 过轻
18.5 ≤ BMI < 24.9: 正常
25 ≤ BMI < 29.9: 超重
BMI ≥ 30: 肥胖
```

```py
import math 

# height = float(input())
# weight = float(input())

def match_bmi(height,weight):
    bmi = weight / math.pow(height,2)
    match bmi :
        case _ if bmi < 18.5 : return "过轻"
        case _ if 18.5 <= bmi < 24.9 : return "正常"
        case _ if 25 <= bmi < 29.9 : return "超重"
        case _ if bmi >= 30 : return "肥胖"

print(match_bmi(160,48))
```
