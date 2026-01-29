---
layout: post
title:  data visualization seaborn 入门-2
date:   2026-01-30 09:01:00 +0800
image: 15.jpg
tags: 
    - python
    - visualization
---

## 直方图 Histogram

```python
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("tips")

sns.histplot(data=df, x="total_bill")
plt.show()
```

* 集中区间
* 分布形状
* 是否右偏 / 左偏

---

### 调整分箱（bins）

```python
sns.histplot(data=df, x="total_bill", bins=20)
plt.show()
```

* bins 太少 → 模糊
* bins 太多 → 噪音

---

## KDE：平滑后的分布

```python
sns.kdeplot(data=df, x="total_bill")
plt.show()
```

KDE 是：

* 概率密度估计
* 「趋势版」直方图

---

### Histogram + KDE（最常用）

```python
sns.histplot(data=df, x="total_bill", kde=True)
plt.show()
```

---

## 分组分布

### 不同性别的账单分布

```python
sns.histplot(
    data=df,
    x="total_bill",
    hue="sex",
    kde=True
)
plt.show()
```

* 谁的消费更分散？
* 是否重叠？

---

## 箱型图 Boxplot

* 中位数在哪？
* 分布范围？
* 有没有异常值？

```python
sns.boxplot(data=df, x="day", y="total_bill")
plt.show()
```

### Boxplot 能告诉你：

* 中位数（不是平均）
* 四分位数
* 离群点（outliers）

---

## 小提琴图 Violin Plot

```python
sns.violinplot(data=df, x="day", y="total_bill")
plt.show()
```

Violin = **Boxplot + KDE**

适合：

* 对比分布形状
* 学术 / 报告展示

---

### 加分组（hue）

```python
sns.violinplot(
    data=df,
    x="day",
    y="total_bill",
    hue="sex",
    split=True
)
plt.show()
```

split=True = 左右对比

---

## 四种分布图怎么选？

| 场景    | 推荐           |
| ----- | ------------ |
| 看整体分布 | hist / kde   |
| 比较群体  | box / violin |
| 找异常   | boxplot      |
| 看形状   | violin       |