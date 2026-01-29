---
layout: post
title:  data visualization seaborn 入门
date:   2026-01-29 09:01:00 +0800
image: 15.jpg
tags: 
    - python
    - visualization
---

# 初次使用

```bash
pip install seaborn
```

```python
import seaborn as sns
import matplotlib.pyplot as pt
import pandas as pd
```

---

# 使用内建数据集

```py
df = sns.load_dataset("tips")
print(df.head())
```

数据包含：

- total_bill（账单）
- tip（小费）
- sex（性别）
- smoker（是否吸烟）
- day（星期）
- time（午/晚餐）

## 折线图

```py
sns.lineplot(data=df, x="total_bill", y="tip")
plt.show()
```

## 柱状图

```py
sns.barplot(
    data=df,
    x="day",
    y="total_bill",
    hue="sex"
)
```

## 点图

```py
sns.pointplot(data=df, x="day", y="tip")
```

---

# 样式

```py
sns.set_theme(style="whitegrid")
```