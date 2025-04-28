---
layout: post
title: python 数字猜谜游戏 专案
date: 2025-02-26 10:24:29 +0800
categories:
    - python
---

任务：  
- 程序随机生成 1-100 之间的一个整数。  
- 让用户猜测数字，并提示 "太大了" 或 "太小了"，直到用户猜对。  
- 输出用户猜测的次数。  

示例输出：  
```
猜一个 1 到 100 之间的数字：50
太小了！
猜一个 1 到 100 之间的数字：75
太大了！
猜一个 1 到 100 之间的数字：62
恭喜！你猜对了，答案是 62，一共猜了 3 次。
```

提示： 使用 `random.randint(1, 100)` 生成随机数。

```py
import random

def match_num(num,count,real_num):
    match num :
        case _ if num > real_num : 
            return "太大了！"
        case _ if num < real_num:
            return "太小了！"
        case real_num : 
            return f"猜对了,{count}"
print(real_num := random.randint(1, 100))        
count = 0

while True : 
    num = float(input())
    print(match_num(num,count,real_num))
    count+=1
```