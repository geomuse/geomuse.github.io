---
layout: post
title: python format
date: 2024-09-29 10:24:29 +0800
categories:
    - python
---

```py
table = {'Sjoerd': 4127, 'Jack': 4098, 'Dcab': 7678}

for name, phone in table.items():
    print(f'{name:10} ==> {phone:10d}')
```