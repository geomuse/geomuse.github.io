---
layout: post
title: python zip
date: 2024-09-21 10:24:29 +0800
categories:
    - python
---

```py
names = ['Cecilla','Lise','marie']
letters = [len(n) for n in names]

for name , count in zip(names,letters) :
    if count > 0 :
        print(name,count)
```