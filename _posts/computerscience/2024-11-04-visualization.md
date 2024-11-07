---
layout: post
title : visualization
date : 2024-11-04 11:24:29 +0800
categories: 
    - stats
    - visualization
---

统一设定

```py
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")
data = sns.load_dataset("tips")["total_bill"]
```

- ### 分布图（Distribution Plot）
    
    `sns.distplot()`: 绘制数据的分布图（包含直方图和核密度估计曲线），用于观察数据的分布特征。

    ![Image Description](/assets/images/output5.png)

    ```py
    sns.distplot(data)
    ```
    
    `sns.kdeplot()`: 核密度估计图，用于显示数据的概率密度。

    ![Image Description](/assets/images/output6.png)

    ```py
    sns.kdeplot(data,color="black")
    ```
    
    `sns.histplot()`: 仅绘制直方图，可以控制柱的宽度、数量等。

    ![Image Description](/assets/images/output7.png)

    ```py
    sns.histplot(data)
    ```

- ### 散点图（Scatter Plot）
    
    `sns.scatterplot()`: 用于展示两变量之间的关系，适用于连续型数据。
    
    `sns.jointplot()`: 结合散点图和分布图，展示两个变量的关系及其各自的分布。
    
    `sns.pairplot()`: 用于探索数据集中所有数值列之间的关系，生成成对的散点图。

- ### 类别图（Categorical Plot）
    
    `sns.barplot()`: 条形图，展示类别数据的平均值或总和。
    
    `sns.boxplot()`: 箱型图，用于显示数据的分布情况及其离散值。
    
    `sns.violinplot()`: 小提琴图，是箱型图的扩展，用于展示数据分布的整体形态。

- ### 线图（Line Plot）
    
    `sns.lineplot()`: 用于展示数据的趋势，尤其适合时间序列数据。

- ### 热图（Heatmap）
    
    `sns.heatmap()`: 通常用于展示二维数据的矩阵热力图，如相关系数矩阵。

- ### 回归图（Regression Plot）
    
    `sns.regplot()`: 绘制带有回归线的散点图，用于展示数据之间的线性关系。