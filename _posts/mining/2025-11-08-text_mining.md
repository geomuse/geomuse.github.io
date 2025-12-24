---
layout: post
title:  python文字探勘 字符编码与文本清理基础
date:   2025-11-08 09:01:00 +0800
tags: 
    - python
    - mining
image: 10.jpg
---

```py
import re

def basic_text_cleaning(text):
    """执行基础文本清理操作：大小写、数字、符号、标点、空白。"""
    
    # 将所有字符转换为小写，实现标准化
    text = text.lower()
    
    # 移除特殊符号和标点符号
    # 使用正则表达式，匹配并替换所有非字母、非数字、非中文、非空白字符
    # [^a-zA-Z0-9\s\u4e00-\u9fa5] 匹配所有不属于这些类别的字符
    # \u4e00-\u9fa5 是中文汉字的 Unicode 范围
    text = re.sub(r'[^a-zA-Z0-9\s\u4e00-\u9fa5]', ' ', text)
    
    # 统一处理多余的空白字符
    # 将多个连续的空白字符（空格、换行、Tab）替换为单个空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

raw_text = """
我是 杰米 AI，今天的日期是 2025/11/07。
WOW! Text Mining 很有用, 价格是 $1999.00！ 
你知道吗…… (符号) *&^%^。 
请检查我的邮件地址：Gemini@google.com。
"""

print("\n--- 原始文本 ---")
print(raw_text)

cleaned_text = basic_text_cleaning(raw_text)

print("\n--- 清理后的文本 ---")
print(cleaned_text)
```

输出结果 

```bash
--- 原始文本 ---

我是 杰米 AI，今天的日期是 2025/11/07。
WOW! Text Mining 很有用, 价格是 $1999.00！ 
你知道吗…… (符号) *&^%^。 
请检查我的邮件地址：Gemini@google.com。


--- 清理后的文本 ---
我是 杰米 ai 今天的日期是 2025 11 07 wow text mining 很有用 价格是 1999 00 你知道吗 符号 请检查我的邮件地址 gemini google com
```