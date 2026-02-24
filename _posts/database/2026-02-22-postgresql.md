---
layout: post
title:  数据库管理postgresql with
date:   2026-02-22 09:01:00 +0800
image: 09.jpg
tags: 
    - postgresql
    - database
---

# 基础用法 (Standard CTE)

基础 CTE 最常用于替代复杂的子查询。

```sql
WITH cte_name AS (
    -- 这里是你的临时查询逻辑
    SELECT column1, column2 FROM table_name WHERE condition
)
SELECT * FROM cte_name; -- 在主查询中像使用表一样使用它
```

# 实际例子

假设我们有一张 orders 表，想要找出那些平均订单金额高于 1000 元的客户及其总消费额。
SQL

```sql
WITH customer_stats AS (
    SELECT 
        customer_id, 
        SUM(amount) AS total_sales,
        AVG(amount) AS avg_amount
    FROM orders
    GROUP BY customer_id
)
SELECT 
    c.customer_name, 
    s.total_sales
FROM customers c
JOIN customer_stats s ON c.id = s.customer_id
WHERE s.avg_amount > 1000;
```