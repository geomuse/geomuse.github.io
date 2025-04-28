---
layout: post
title: python 统计字符串中的字母、数字、空格和其他字符个数 专案
date: 2025-02-28 10:24:29 +0800
categories:
    - python
---

任务：  
- 让用户输入一个字符串，统计其中：
  - 英文字母个数
  - 数字个数
  - 空格个数
  - 其他字符个数  

示例输入：  
```
请输入一串字符：Hello 123!
```
示例输出：  
```
字母个数：5
数字个数：3
空格个数：1
其他字符个数：1
```

提示： 使用 `isalpha()`、`isdigit()` 和 `isspace()` 方法判断字符类型。

```py
import re

def count_chars(s):
    digits = len(re.findall(r'\d', s))  # 匹配数字
    letters = len(re.findall(r'[a-zA-Z]', s))  # 匹配英文字母
    chinese = len(re.findall(r'[\u4e00-\u9fff]', s))  # 匹配中文字符
    return {"数字": digits, "英文字母": letters, "中文字符": chinese}

# 测试
s = "Python 3.8 版本包含了一些新特性"
result = count_chars(s)
print(result)
```

```bash
{'数字': 2, '英文字母': 6, '中文字符': 8}
```