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