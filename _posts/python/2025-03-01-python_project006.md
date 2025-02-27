---
layout: post
title: python 最大值函数 专案
date: 2025-03-01 10:24:29 +0800
categories:
    - python
---

任务：  
- 编写一个函数 `find_max(num1, num2, num3)`，返回三个数中的最大值。  
- 要求使用 `if-else` 语句。  

示例输入：  
```python
print(find_max(5, 12, 9))
```
示例输出：  
```
12
```

```py
def find_max(num1,num2,num3):
    r = num1 
    z = num1 , num2 , num3
    for num in z :
        if num > r :
            r = num
    return r
    
print(find_max(5,12,90))
```