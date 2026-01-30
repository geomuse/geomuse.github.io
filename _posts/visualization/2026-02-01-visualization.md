---
layout: post
title:  data visualization seaborn 入门-4
date:   2026-02-01 09:01:00 +0800
image: 15.jpg
tags: 
    - python
    - visualization
---

# Heatmap

```py
corr = df.corr(numeric_only=True)
sns.heatmap(corr)
```

# 热力图

加数值 + 调色

```py
sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)
```

# 处理信息过载（遮住一半）

```py
mask = np.triu(corr)

sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    cmap="coolwarm"
)
```

只看下三角

商业报表常用