---
layout: post
title:  数据库管理postgresql 新增数据
date:   2025-10-06 09:01:00 +0800
image: 09.jpg
tags: 
    - python
    - postgresql
---

理解表结构的关键在于把握两个核心要素：**数据类型 (Data Types)** 和 **约束 (Constraints)**。它们共同决定了数据在数据库中的存储方式、格式和有效性。

### 核心数据类型详解 (Data Types)

数据类型定义了字段可以存储何种类型的数据（文本、数字、日期、布尔值等）。PostgreSQL 以其丰富且强大的数据类型而闻名。

| 类型分类 | 常用数据类型 | 描述 | 适用场景 |
| :--- | :--- | :--- | :--- |
| **整数** | **`SMALLINT`** | 2 字节，范围较小。 | 小计数器、状态码。 |
| | **`INTEGER` (INT)** | 4 字节，常用整数。 | 大多数数字字段，如年龄、数量。 |
| | **`BIGINT`** | 8 字节，范围极大。 | 大容量计数器、高并发 ID。 |
| | **`SERIAL`** | 自动递增的 `INTEGER`。 (如您 `id` 字段) | 主键 (Primary Key) ID。 |
| | **`BIGSERIAL`** | 自动递增的 `BIGINT`。 | 大表的主键 ID。 |
| **浮点数** | **`REAL`** | 4 字节单精度浮点数。 | 科学计算、对精度要求不高的场景。 |
| | **`DOUBLE PRECISION`** | 8 字节双精度浮点数。 | 更精确的浮点计算。 |
| | **`NUMERIC(p, s)`** | **精确**数值类型。`p` 是总位数，`s` 是小数位数。 | 货币金额、汇率、精确计算。 |
| **字符/文本** | **`VARCHAR(n)`** | 可变长字符串，限制最大长度 `n`。 | 姓名、地址等有长度限制的字段。 |
| | **`TEXT`** | 可变长字符串，**无最大限制**。 (如您 `name` 字段) | 文章内容、长描述、大段 JSON/XML 文本。 |
| **布尔** | **`BOOLEAN`** | 只能存储 `TRUE`, `FALSE`, 或 `NULL`。 | 表示开关、状态等二值逻辑。 |
| **日期/时间** | **`DATE`** | 仅存储日期 (年、月、日)。 | 生日、事件发生日期。 |
| | **`TIMESTAMP`** | 存储日期和时间，**不含**时区信息。 | 记录操作时间。 |
| | **`TIMESTAMP WITH TIME ZONE`** | 存储日期和时间，**包含**时区信息。 | **强烈推荐**，确保跨地域/应用时间一致性。 |
| **高级类型** | **`JSONB`** | 二进制 JSON 格式，支持索引和高效查询。 | 存储非结构化数据或配置信息。 |
| | **`UUID`** | 通用唯一标识符 (Universally Unique Identifier)。 | 分布式系统中用作主键。 |

-----

### 核心约束详解 (Constraints)

约束是数据库用来保证数据完整性和有效性的规则。它们限制了可以插入到表中的数据。

| 约束类型 | 语法示例 | 描述 | 作用 |
| :--- | :--- | :--- | :--- |
| **`PRIMARY KEY`** | `id SERIAL PRIMARY KEY` | **主键。** 是 **`NOT NULL`** 和 **`UNIQUE`** 的组合。 | **唯一且非空**地标识表中的每一行，是表的核心。 |
| **`NOT NULL`** | `email TEXT NOT NULL` | **非空。** | 强制该字段在插入数据时必须有值，不能为 `NULL`。 |
| **`UNIQUE`** | `username TEXT UNIQUE` | **唯一。** | 强制该字段（或一组字段）在整个表中不能有重复值。 |
| **`FOREIGN KEY`** | `user_id INT REFERENCES users(id)` | **外键。** | 建立两个表之间的关联。确保该字段的值必须在引用的父表字段中存在，维护了**参照完整性**。 |
| **`CHECK`** | `age INT CHECK (age > 0)` | **检查。** | 允许您定义一个自定义的布尔表达式，只有满足表达式的行才能插入。 |
| **`DEFAULT`** | `status TEXT DEFAULT 'active'` | **默认值。** (严格来说不是约束) | 如果插入新行时没有为该字段提供值，则自动使用预设的值。 |

### 示例扩展您的 `users` 表

如果您想创建一个更健壮的 `users` 表结构，可以结合这些知识：

```sql
CREATE TABLE users(id SERIAL PRIMARY KEY, name TEXT, age INT);
```

```sql
CREATE TABLE users_extended (
    id BIGSERIAL PRIMARY KEY,  -- 使用 BIGSERIAL 应对大数据量
    username VARCHAR(50) UNIQUE NOT NULL, -- 强制用户名唯一且非空
    email VARCHAR(100) UNIQUE,           -- 强制邮箱唯一 (允许为 NULL)
    age INTEGER CHECK (age >= 18),       -- 检查年龄必须大于等于 18 岁
    status VARCHAR(10) DEFAULT 'pending', -- 默认状态为 'pending'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() -- 自动记录创建时间，带时区
);
```