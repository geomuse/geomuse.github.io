---
layout: post
title : violin plot
date : 2024-10-31 11:24:29 +0800
categories: 
    - stats
    - visualization
---

<script>
  MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']]
    }
  };
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

小提琴图（Violin Plot）是一种用于可视化数据分布的统计图，结合了箱线图和密度图的特点。它的设计在展示数据的分布形状时具有较强的直观性，特别适用于观察数据的集中趋势和离散情况。
小提琴图的特点

    数据分布：小提琴图通过在两侧显示数据的核密度估计（Kernel Density Estimation, KDE）曲线，表示数据的分布。图形的宽度越宽，表示数据的密度越高。
    内嵌的箱线图：通常，小提琴图内部包含一个箱线图，显示数据的中位数、四分位数以及上下须等，帮助了解数据的分布趋势和离散程度。

    对称性：小提琴图的形状通常是对称的，每侧的形状是数据分布的镜像，有助于更清晰地观察数据的分布形态。

何时使用小提琴图？

小提琴图特别适用于多组数据对比的场景。比如，比较不同分类变量的数值分布，或多个组的分布形态差异（如不同类别的考试成绩分布）。它能够比箱线图提供更丰富的分布信息，适合数据样本较大且分布不完全对称的情况。
小提琴图的优缺点
优点：

    更清晰展示分布形态和密度，适用于展示数据集中趋势及分散程度。
    
    能够显示数据是否对称、是否存在多峰分布（多峰意味着数据有多个密集区域）。

缺点：

    不适合样本量过少的数据，因为密度估计可能不准确，导致图形失真。
    
    复杂度较高，不如箱线图简单明了，因此在解释上有一定门槛。

小提琴图通过视觉上增强的方式展示数据分布，是探索性数据分析中的一项有力工具。

![Image Description](/assets/images/output.png)

```py
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")

# Load the example dataset of brain network correlations
df = sns.load_dataset("brain_networks", header=[0, 1, 2], index_col=0)

# Pull out a specific subset of networks
used_networks = [1, 3, 4, 5, 6, 7, 8, 11, 12, 13, 16, 17]
used_columns = (df.columns.get_level_values("network")
                          .astype(int)
                          .isin(used_networks))
df = df.loc[:, used_columns]

# Compute the correlation matrix and average over networks
corr_df = df.corr().groupby(level="network").mean()
corr_df.index = corr_df.index.astype(int)
corr_df = corr_df.sort_index().T

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 6))

# Draw a violinplot with a narrower bandwidth than the default
sns.violinplot(data=corr_df, bw_adjust=.5, cut=1, linewidth=1, palette="Set3")

# Finalize the figure
ax.set(ylim=(-.7, 1.05))
sns.despine(left=True, bottom=True)
```