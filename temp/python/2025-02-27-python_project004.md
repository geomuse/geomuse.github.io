---
layout: post
title: python 打印乘法表（九九乘法表） 专案
date: 2025-02-27 10:24:29 +0800
categories:
    - python
---

任务：  
- 用 嵌套循环 输出 1-9 乘法表。  

示例输出：  
```
1 x 1 = 1
1 x 2 = 2  2 x 2 = 4
1 x 3 = 3  2 x 3 = 6  3 x 3 = 9
...
```

提示： 使用 `for` 循环嵌套。

```py
for _ in range(1,12+1):
    for __ in range(1,12+1):
        print(f'{_}*{__}={_*__}')
```