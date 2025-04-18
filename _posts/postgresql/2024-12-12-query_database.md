---
layout: post
title:  postgresql 基本查询 database
date:   2024-12-12 11:24:29 +0800
categories: 
    - database
    - postgresql
---

#### **1. 选择特定列**
在 PostgreSQL 中，你可以使用 `SELECT` 语句从表中检索特定列。

**语法**：
```sql
SELECT column1, column2
FROM table_name;
```

**示例**：
假设有一个名为 `employees` 的表：
```sql
SELECT name, position
FROM employees;
```

结果将返回 `name` 和 `position` 两列。

#### **2. 添加条件：`WHERE` 子句**
`WHERE` 子句用于筛选符合条件的行。

**语法**：
```sql
SELECT column1, column2
FROM table_name
WHERE condition;
```

**示例**：
获取薪资大于 4000 的员工：
```sql
SELECT name, salary
FROM employees
WHERE salary > 4000;
```

**支持的条件运算符**：
- 比较运算符：`=`, `!=`, `>`, `<`, `>=`, `<=`
- 逻辑运算符：`AND`, `OR`, `NOT`
- 范围：`BETWEEN min_value AND max_value`
- 集合：`IN (value1, value2, ...)`
- 模糊匹配：`LIKE 'pattern'`（如 `%` 表示任意多个字符，`_` 表示一个字符）

**示例**：
筛选职位为 "Manager" 且薪资高于 5000 的员工：
```sql
SELECT name, salary
FROM employees
WHERE position = 'Manager' AND salary > 5000;
```

#### **3. 排序：`ORDER BY` 子句**
`ORDER BY` 用于对查询结果按指定列排序。

**语法**：
```sql
SELECT column1, column2
FROM table_name
ORDER BY column1 [ASC|DESC];
```
- 默认是升序（`ASC`），降序需要显式指定 `DESC`。

**示例**：
按薪资从高到低排序：
```sql
SELECT name, salary
FROM employees
ORDER BY salary DESC;
```

按多个列排序：
```sql
SELECT name, position, salary
FROM employees
ORDER BY position ASC, salary DESC;
```

#### **4. 限制行数：`LIMIT` 和 `OFFSET`**
- **`LIMIT`**：限制返回的行数。
- **`OFFSET`**：跳过指定的行数。

**语法**：
```sql
SELECT column1, column2
FROM table_name
ORDER BY column_name
LIMIT row_count OFFSET start_point;
```

**示例**：
获取薪资最高的 3 名员工：
```sql
SELECT name, salary
FROM employees
ORDER BY salary DESC
LIMIT 3;
```

获取薪资最高的第 4 到第 6 名员工：
```sql
SELECT name, salary
FROM employees
ORDER BY salary DESC
LIMIT 3 OFFSET 3;
```

### 综合练习
假设 `employees` 表的内容如下：
| id | name       | position   | salary | hire_date  |
|----|------------|------------|--------|------------|
| 1  | John Doe   | Manager    | 6000   | 2022-01-15 |
| 2  | Jane Smith | Developer  | 5000   | 2021-03-12 |
| 3  | Alice Brown| Analyst    | 4000   | 2020-07-18 |
| 4  | Bob White  | Intern     | 2000   | 2023-02-25 |
| 5  | Emma Davis | Developer  | 4500   | 2021-10-05 |

- 获取薪资高于 4000 的员工名字和职位，按薪资降序排列：
  ```sql
  SELECT name, position
  FROM employees
  WHERE salary > 4000
  ORDER BY salary DESC;
  ```

- 获取薪资最低的两名员工：
  ```sql
  SELECT name, salary
  FROM employees
  ORDER BY salary ASC
  LIMIT 2;
  ```

- 获取从第 3 名开始的薪资最高的 2 名员工：
  ```sql
  SELECT name, salary
  FROM employees
  ORDER BY salary DESC
  LIMIT 2 OFFSET 2;
  ```

希望这些内容能够帮助你熟悉 PostgreSQL 的基本查询！如果有进一步问题，可以随时询问。