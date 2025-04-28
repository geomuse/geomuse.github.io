---
layout: post
title : distribution
date : 2024-10-31 11:24:29 +0800
categories: 
    - stats
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

统计学中的分布种类繁多，每种分布在不同的研究领域和统计分析中都具有特定的应用。以下是主要的分布类型和简要描述：
离散分布

- 二项分布（Binomial Distribution）：描述一组独立的伯努利试验的成功次数。
- 泊松分布（Poisson Distribution）：用于描述在固定时间内发生的事件数，适用于稀有事件。
- 几何分布（Geometric Distribution）：描述直到第一次成功前的失败次数。
- 负二项分布（Negative Binomial Distribution）：类似于几何分布，表示在指定成功次数之前的失败次数。
- 超几何分布（Hypergeometric Distribution）：用于描述在无放回抽样中，抽到特定类别对象的次数。

连续分布

- 正态分布（Normal Distribution）：也称高斯分布，许多自然现象符合该分布。
- 均匀分布（Uniform Distribution）：每个区间内的值出现概率相同。
- 指数分布（Exponential Distribution）：描述事件发生的时间间隔，常用于生存分析。
- 卡方分布（Chi-Square Distribution）：广泛用于统计检验。
- t分布（t-Distribution）：用于小样本均值的显著性检验。
- F分布（F-Distribution）：用于方差分析（ANOVA）和回归分析中的方差比较。
- 贝塔分布（Beta Distribution）：定义在[0,1]区间内，适用于概率参数的分布。
- 伽马分布（Gamma Distribution）：广泛用于等待时间和生存分析。
- 威布尔分布（Weibull Distribution）：用于可靠性和寿命数据分析。
- 柯西分布（Cauchy Distribution）：重尾分布，没有均值和方差的定义。

特殊分布

- 拉普拉斯分布（Laplace Distribution）：类似于正态分布，但具有较厚的尾部。
- 对数正态分布（Log-Normal Distribution）：当变量的对数服从正态分布时，原变量服从对数正态分布。
- 帕累托分布（Pareto Distribution）：用于描述财富分布和自然现象。
- 柯尔莫戈罗夫分布（Kolmogorov Distribution）：用于分布拟合优度检验。

常用的复合分布和混合分布

- 正态-逆伽马分布（Normal-Inverse Gamma Distribution）：常用于贝叶斯推断中。
- 混合正态分布（Mixture of Normals）：用于建模复杂的数据分布，通过多个正态分布组合而成。

![Image Description](/assets/images/Figure_1.png)


```py
import numpy as np
import matplotlib.pyplot as pt
import scipy.stats as stats
from matplotlib import style
style.use('ggplot')

# 设置图形布局
fig, axs = pt.subplots(5, 3, figsize=(15, 20))
fig.suptitle("Major Statistical Distributions", fontsize=20)
pt.subplots_adjust(hspace=0.5, wspace=0.3)

# 定义分布的名称、数据和设置
distributions = [
    ('Binomial', np.arange(0, 20), stats.binom.pmf(np.arange(0, 20), n=20, p=0.5), "Binomial Distribution"),
    ('Poisson', np.arange(0, 20), stats.poisson.pmf(np.arange(0, 20), mu=5), "Poisson Distribution"),
    ('Geometric', np.arange(1, 20), stats.geom.pmf(np.arange(1, 20), p=0.5), "Geometric Distribution"),
    ('Negative Binomial', np.arange(0, 20), stats.nbinom.pmf(np.arange(0, 20), n=5, p=0.5), "Negative Binomial Distribution"),
    ('Hypergeometric', np.arange(0, 10), stats.hypergeom.pmf(np.arange(0, 10), 20, 7, 12), "Hypergeometric Distribution"),
    ('Normal', np.linspace(-4, 4, 100), stats.norm.pdf(np.linspace(-4, 4, 100), loc=0, scale=1), "Normal Distribution"),
    ('Uniform', np.linspace(0, 1, 100), stats.uniform.pdf(np.linspace(0, 1, 100)), "Uniform Distribution"),
    ('Exponential', np.linspace(0, 4, 100), stats.expon.pdf(np.linspace(0, 4, 100)), "Exponential Distribution"),
    ('Chi-Square', np.linspace(0, 10, 100), stats.chi2.pdf(np.linspace(0, 10, 100), df=2), "Chi-Square Distribution"),
    ('t-Distribution', np.linspace(-4, 4, 100), stats.t.pdf(np.linspace(-4, 4, 100), df=10), "t-Distribution"),
    ('F-Distribution', np.linspace(0, 4, 100), stats.f.pdf(np.linspace(0, 4, 100), dfn=5, dfd=2), "F Distribution"),
    ('Beta', np.linspace(0, 1, 100), stats.beta.pdf(np.linspace(0, 1, 100), a=2, b=5), "Beta Distribution"),
    ('Gamma', np.linspace(0, 10, 100), stats.gamma.pdf(np.linspace(0, 10, 100), a=2), "Gamma Distribution"),
    ('Weibull', np.linspace(0, 2, 100), stats.weibull_min.pdf(np.linspace(0, 2, 100), c=1.5), "Weibull Distribution"),
    ('Cauchy', np.linspace(-4, 4, 100), stats.cauchy.pdf(np.linspace(-4, 4, 100)), "Cauchy Distribution")
]

# 绘制每个分布
for ax, (name, x, y, title) in zip(axs.ravel(), distributions):
    ax.plot(x, y, label=title)
    ax.set_title(title)
    ax.legend()
    ax.grid()

pt.show()
```