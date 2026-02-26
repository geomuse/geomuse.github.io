---
layout: post
title:  python文字探勘 链式表达式 (Method Chaining)
date:   2026-02-25 09:01:00 +0800
tags: 
    - python
    - mining
image: 10.jpg
---

```py
import polars as pl

# 模拟一份评论数据
data = {
    "user_id": [1, 2, 3, 4],
    "comment": ["Very good product!", "Bad service, slow delivery.", "Excellent!", "Average, but okay."]
}
df = pl.DataFrame(data)

# 高级特征提取
featured_df = df.with_columns([
    # 1. 计算字符长度
    pl.col("comment").str.len_chars().alias("char_count"),
    
    # 2. 计算单词数量 (按空格切分并计算长度)
    pl.col("comment").str.split(" ").list.len().alias("word_count"),
    
    # 3. 提取特定关键词（如：是否包含 'good' 或 'excellent'）
    pl.col("comment").str.to_lowercase().str.contains(r"good|excellent").alias("is_positive")
])

print(featured_df)
```

```bash
shape: (4, 5)
┌─────────┬─────────────────────────────┬────────────┬────────────┬─────────────┐
│ user_id ┆ comment                     ┆ char_count ┆ word_count ┆ is_positive │
│ ---     ┆ ---                         ┆ ---        ┆ ---        ┆ ---         │
│ i64     ┆ str                         ┆ u32        ┆ u32        ┆ bool        │
╞═════════╪═════════════════════════════╪════════════╪════════════╪═════════════╡
│ 1       ┆ Very good product!          ┆ 18         ┆ 3          ┆ true        │
│ 2       ┆ Bad service, slow delivery. ┆ 27         ┆ 4          ┆ false       │
│ 3       ┆ Excellent!                  ┆ 10         ┆ 1          ┆ true        │
│ 4       ┆ Average, but okay.          ┆ 18         ┆ 3          ┆ false       │
└─────────┴─────────────────────────────┴────────────┴────────────┴─────────────┘
```