---
layout: post
title:  postgresql 数据汇总和计算
date:   2024-12-14 11:24:29 +0800
categories: 
    - database
    - postgresql
---

### **聚合函数简介**

| 聚合函数  | 说明                           |
|-----------|-------------------------------|
| `COUNT`   | 计算行数或非空值的数量          |
| `SUM`     | 求和                           |
| `AVG`     | 计算平均值                     |
| `MIN`     | 获取最小值                     |
| `MAX`     | 获取最大值                     |

---

### **聚合函数的用法**

#### **`COUNT`**
计算行数或某列的非空值数量。  
```sql
-- 统计订单总数
SELECT COUNT(*) AS total_orders
FROM orders;

-- 统计有具体金额的订单数量
SELECT COUNT(order_amount) AS non_null_orders
FROM orders;
```

#### **`SUM`**
对某列的数值求和。  
```sql
-- 统计订单总金额
SELECT SUM(order_amount) AS total_revenue
FROM orders;
```

#### **`AVG`**
计算某列的平均值。  
```sql
-- 计算订单的平均金额
SELECT AVG(order_amount) AS average_order
FROM orders;
```

#### **`MIN` 和 `MAX`**
获取某列的最小值和最大值。  
```sql
-- 获取订单的最低和最高金额
SELECT MIN(order_amount) AS minimum_order,
       MAX(order_amount) AS maximum_order
FROM orders;
```

---

### **使用 `GROUP BY` 分组**

`GROUP BY` 用于将查询结果按某列分组，并对每组应用聚合函数。

#### **示例**
1. 统计每位客户的订单总金额：
   ```sql
   SELECT customer_id, SUM(order_amount) AS total_spent
   FROM orders
   GROUP BY customer_id;
   ```

2. 统计每个类别的商品数量：
   ```sql
   SELECT category, COUNT(*) AS product_count
   FROM products
   GROUP BY category;
   ```

---

### **使用 `HAVING` 子句**

`HAVING` 子句用于过滤分组后的数据，类似于 `WHERE`，但只能用于聚合函数。

#### **示例**
1. 筛选消费总额大于 1000 的客户：
   ```sql
   SELECT customer_id, SUM(order_amount) AS total_spent
   FROM orders
   GROUP BY customer_id
   HAVING SUM(order_amount) > 1000;
   ```

2. 筛选订单数量大于 3 的客户：
   ```sql
   SELECT customer_id, COUNT(*) AS order_count
   FROM orders
   GROUP BY customer_id
   HAVING COUNT(*) > 3;
   ```

---

### **综合案例**

#### **案例：订单分析**

假设有以下两张表：
- `customers` 表：
  | customer_id | name         |
  |-------------|--------------|
  | 1           | Alice        |
  | 2           | Bob          |
  | 3           | Charlie      |

- `orders` 表：
  | order_id | customer_id | order_amount |
  |----------|-------------|--------------|
  | 1        | 1           | 500          |
  | 2        | 1           | 300          |
  | 3        | 2           | 700          |
  | 4        | 3           | 200          |

#### **任务**
1. 查询每位客户的订单总金额：
   ```sql
   SELECT customers.name, SUM(orders.order_amount) AS total_spent
   FROM customers
   JOIN orders ON customers.customer_id = orders.customer_id
   GROUP BY customers.name;
   ```

   **结果**：
   | name    | total_spent |
   |---------|-------------|
   | Alice   | 800         |
   | Bob     | 700         |
   | Charlie | 200         |

2. 查询订单总金额超过 500 的客户：
   ```sql
   SELECT customers.name, SUM(orders.order_amount) AS total_spent
   FROM customers
   JOIN orders ON customers.customer_id = orders.customer_id
   GROUP BY customers.name
   HAVING SUM(orders.order_amount) > 500;
   ```

   **结果**：
   | name    | total_spent |
   |---------|-------------|
   | Alice   | 800         |
   | Bob     | 700         |

---

### **练习任务**

1. 创建以下两张表：
   ```sql
   CREATE TABLE sales (
       id INT PRIMARY KEY,
       product_name VARCHAR(50),
       category VARCHAR(50),
       amount DECIMAL(10, 2)
   );

   CREATE TABLE employees (
       id INT PRIMARY KEY,
       name VARCHAR(50),
       department VARCHAR(50),
       salary DECIMAL(10, 2)
   );
   ```

2. 插入数据并完成以下查询：
   - 按 `category` 统计商品的销售总额。
   - 筛选销售额超过 1000 的类别。
   - 查询每个部门的平均薪资，并筛选出高于 5000 的部门。
