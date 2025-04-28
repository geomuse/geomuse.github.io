---
layout: post
title:  postgresql 子查询
date:   2024-12-15 11:24:29 +0800
categories: 
    - database
    - postgresql
---

### **Day 10: 子查询**

子查询（Subquery）是嵌套在另一条 SQL 查询语句中的查询，通常用来为主查询提供数据。子查询可用于 `SELECT`、`FROM`、`WHERE`、`HAVING` 等子句中。

---

### **1. 简单子查询**
简单子查询是最常见的形式，用于提供数据给主查询。

#### **示例：**
```sql
-- 找到销售额大于平均销售额的订单
SELECT order_id, amount
FROM orders
WHERE amount > (
    SELECT AVG(amount)
    FROM orders
);
```

#### **解析：**
- 子查询：`SELECT AVG(amount) FROM orders` 返回 `orders` 表的平均销售额。
- 主查询：从 `orders` 表中筛选出大于该平均值的订单。

---

### **2. 子查询作为表的一部分**
子查询也可以出现在 `FROM` 子句中，作为临时表。

#### **示例：**
```sql
-- 计算每个订单的排名
SELECT sub.order_id, sub.amount, RANK() OVER (ORDER BY sub.amount DESC) AS rank
FROM (
    SELECT order_id, amount
    FROM orders
) sub;
```

#### **解析：**
- 子查询：`SELECT order_id, amount FROM orders` 创建了一个临时表 `sub`。
- 主查询：在 `sub` 的基础上，计算订单金额的排名。

---

### **3. 相关子查询**
相关子查询（Correlated Subquery）会依赖于主查询中的数据行逐条执行，通常通过 `EXISTS` 或 `NOT EXISTS` 语句表示。

#### **示例：**
```sql
-- 找到至少有一件商品的客户
SELECT customer_id
FROM customers c
WHERE EXISTS (
    SELECT 1
    FROM orders o
    WHERE o.customer_id = c.customer_id
);
```

#### **解析：**
- 主查询：遍历 `customers` 表中的每一行。
- 子查询：检查当前客户是否在 `orders` 表中有对应订单。
- `EXISTS`：如果子查询返回结果，条件成立。

---

### **4. 使用子查询优化复杂查询**
通过将复杂的逻辑分解为子查询，查询变得更清晰和高效。

#### **示例：**
```sql
-- 找到购买金额最高的客户
SELECT customer_id
FROM orders
WHERE amount = (
    SELECT MAX(amount)
    FROM orders
);
```

#### **优化后的多子查询形式：**
```sql
-- 先找出最高金额
WITH max_amount AS (
    SELECT MAX(amount) AS max_amt
    FROM orders
)
SELECT customer_id
FROM orders, max_amount
WHERE orders.amount = max_amount.max_amt;
```

---

### **关键点总结：**
- **简单子查询：** 适合单值返回的情况（如标量子查询）。
- **子查询作为表：** 常用于复杂数据的分解和逻辑的清晰化。
- **相关子查询：** 当子查询需要依赖主查询数据时使用。
- **优化复杂查询：** 使用子查询或 CTE 提高可读性和性能。