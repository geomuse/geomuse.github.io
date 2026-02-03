---
layout: post
title:  data visualization plotly入门
date:   2026-02-03 09:01:00 +0800
image: 15.jpg
tags: 
    - python
    - visualization
---

# 折线图

```py
import plotly.express as px
import pandas as pd

df = px.data.stocks()

fig = px.line(
    df,
    x="date",
    y="GOOG",
    title="Google 股价走势"
)

fig.show()
```

# 交互式散点图

```py
df = px.data.iris()

fig = px.scatter(
    df,
    x="sepal_width",
    y="sepal_length",
    color="species",
    size="petal_length",
    title="鸢尾花特征关系"
)

fig.show()
```