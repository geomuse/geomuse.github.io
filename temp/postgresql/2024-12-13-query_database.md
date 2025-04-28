---
layout: post
title:  postgresql 多表查询 database
date:   2024-12-13 11:24:29 +0800
categories: 
    - database
    - postgresql
---

多表查询是数据库中处理复杂数据结构的关键部分。本节将学习如何通过表与表之间的关系完成高效的数据提取。

---

### **1. 理解表之间的关系**

- **一对一关系**  
  每个表中的一行与另一个表中的一行相关联。  
  **示例**：用户表和用户详情表。

- **一对多关系**  
  一个表中的一行可以与另一个表中的多行相关联。  
  **示例**：客户表和订单表（一个客户可以有多个订单）。

- **多对多关系**  
  一个表中的多行可以与另一个表中的多行相关联，通常需要一个中间表（关联表）。  
  **示例**：学生表和课程表（一个学生可以选多门课，一门课可以被多名学生选）。

---

### **2. JOIN 类型及其用途**

1. **`INNER JOIN`（内连接）**  
   返回两个表中匹配的记录（交集）。  
   ```sql
   SELECT a.column1, b.column2
   FROM TableA a
   INNER JOIN TableB b ON a.common_column = b.common_column;
   ```

2. **`LEFT JOIN`（左连接）**  
   返回左表中的所有记录，即使右表中没有匹配。  
   ```sql
   SELECT a.column1, b.column2
   FROM TableA a
   LEFT JOIN TableB b ON a.common_column = b.common_column;
   ```

3. **`RIGHT JOIN`（右连接）**  
   返回右表中的所有记录，即使左表中没有匹配。  
   ```sql
   SELECT a.column1, b.column2
   FROM TableA a
   RIGHT JOIN TableB b ON a.common_column = b.common_column;
   ```

4. **`FULL OUTER JOIN`（全外连接）**  
   返回两个表中的所有记录，当其中一个表没有匹配时，用 `NULL` 填充。  
   ```sql
   SELECT a.column1, b.column2
   FROM TableA a
   FULL OUTER JOIN TableB b ON a.common_column = b.common_column;
   ```

---

### **3. 实践常见的多表查询场景**

#### **案例 1: INNER JOIN 示例**
- 查询客户的订单信息：  
  ```sql
  SELECT customers.name, orders.order_date
  FROM customers
  INNER JOIN orders ON customers.id = orders.customer_id;
  ```

#### **案例 2: LEFT JOIN 示例**
- 查询所有客户及其订单信息（包括未下单的客户）：  
  ```sql
  SELECT customers.name, orders.order_date
  FROM customers
  LEFT JOIN orders ON customers.id = orders.customer_id;
  ```

#### **案例 3: RIGHT JOIN 示例**
- 查询所有订单及其对应的客户信息（包括找不到客户的订单）：  
  ```sql
  SELECT customers.name, orders.order_date
  FROM customers
  RIGHT JOIN orders ON customers.id = orders.customer_id;
  ```

#### **案例 4: FULL OUTER JOIN 示例**
- 查询所有客户和订单信息（包括没有订单的客户和没有客户信息的订单）：  
  ```sql
  SELECT customers.name, orders.order_date
  FROM customers
  FULL OUTER JOIN orders ON customers.id = orders.customer_id;
  ```

---

### **练习任务**

1. 创建以下两张表：
   ```sql
   CREATE TABLE students (
       id INT PRIMARY KEY,
       name VARCHAR(50)
   );

   CREATE TABLE courses (
       id INT PRIMARY KEY,
       course_name VARCHAR(50)
   );

   CREATE TABLE student_courses (
       student_id INT,
       course_id INT,
       FOREIGN KEY (student_id) REFERENCES students(id),
       FOREIGN KEY (course_id) REFERENCES courses(id)
   );
   ```

2. 插入数据并尝试以下查询：
   - 查询学生及其选修课程名称。
   - 查询所有课程及其选修的学生（包括没有学生选修的课程）。  
   - 查询既不在学生表也不在课程表中的记录。

---

### **总结**
- 理解表关系是 JOIN 的基础。
- 根据需求选择不同的 JOIN 类型：
  - 聚焦于匹配数据用 `INNER JOIN`。
  - 包含左表或右表完整数据用 `LEFT JOIN` 或 `RIGHT JOIN`。
  - 需要完整数据用 `FULL OUTER JOIN`。