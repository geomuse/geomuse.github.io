---
layout: post
title: python 文件写入与读取   专案
date: 2025-03-01 10:24:29 +0800
categories:
    - python
---

任务：  
- 让用户输入一个句子，并将其写入 `output.txt` 文件。  
- 之后，再读取 `output.txt`，并打印其中的内容。  

示例输入：  
```
请输入一句话：Hello, Python!
```
示例输出（读取文件内容）：  
```
文件内容：Hello, Python!
```

提示：  
- 使用 `open('output.txt', 'w', encoding='utf-8')` 写入文件。  
- 使用 `open('output.txt', 'r', encoding='utf-8')` 读取文件。


```py
text = "Python is a great programming language. It is easy to learn and powerful."

with open('output.txt', 'w', encoding='utf-8') as f :
    f.write(text)

with open('output.txt', 'r', encoding='utf-8') as f :
    print(f.readline())
```