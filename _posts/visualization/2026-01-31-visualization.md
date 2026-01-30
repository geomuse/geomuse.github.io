---
layout: post
title:  data visualization seaborn 入门-3
date:   2026-01-31 09:01:00 +0800
image: 15.jpg
tags: 
    - python
    - visualization
---

判断 两个变量有没有关系

用散点图看：

正相关 / 负相关 / 无关

用回归线判断趋势

用 pairplot 快速扫全局

# 散点图 Scatter Plot

```py
sns.scatterplot(
    data=df,
    x="total_bill",
    y="tip",
    hue="sex"
)
```

```py
sns.scatterplot(
    data=df,
    x="total_bill",
    y="tip",
    alpha=0.6,
    hue='sex'
)
```

# 回归图 Regression Plot

```py
sns.regplot(
    data=df,
    x="total_bill",
    y="tip"
)
```

# 按类别分回归

```py
sns.lmplot(
    data=df,
    x="total_bill",
    y="tip",
    hue="sex"
)
```

# 多变量快速扫描：pairplot

```
sns.pairplot(df, hue="sex")
```