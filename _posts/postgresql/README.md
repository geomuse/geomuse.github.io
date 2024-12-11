以下是 **PostgreSQL 30 天学习计划**，帮助你系统地掌握 PostgreSQL 数据库的基础知识和高级技能。

---

### **第 1 周：基础概念与安装**
#### **Day 1: 数据库基础**
- 理解什么是数据库、RDBMS（关系数据库管理系统）。
- 了解 PostgreSQL 的特点与适用场景。
- 阅读官方文档入门部分（[PostgreSQL 官方文档](https://www.postgresql.org/docs/)）。

#### **Day 2: 安装与配置**  <!-- done -->
- 安装 PostgreSQL：
  - Windows、Mac 或 Linux 的安装步骤。
  - 使用 Docker 安装（推荐用于快速测试）。
- 学习 `psql` 命令行工具的基本使用。
- 创建第一个数据库并连接。

#### **Day 3: 数据库与表**  <!-- done -->
- 创建、删除数据库：`CREATE DATABASE` 和 `DROP DATABASE`。
- 创建、修改、删除表：`CREATE TABLE`、`ALTER TABLE` 和 `DROP TABLE`。
- 了解 PostgreSQL 数据类型：`INT`、`VARCHAR`、`TEXT`、`BOOLEAN`、`DATE` 等。

#### **Day 4: 数据操作（CRUD 操作）**  <!-- done -->
- 插入数据：`INSERT INTO`。
- 查询数据：`SELECT`。
- 更新数据：`UPDATE`。
- 删除数据：`DELETE`。

#### **Day 5: 基本查询** <!-- done -->
- 选择特定列：`SELECT column1, column2 FROM table`。
- 添加条件：`WHERE` 子句。
- 排序：`ORDER BY` 子句。
- 限制行数：`LIMIT` 和 `OFFSET`。

#### **Day 6: 索引**
- 理解索引的作用。
- 创建、删除索引：`CREATE INDEX` 和 `DROP INDEX`。
- 了解 PostgreSQL 的索引类型（`B-tree`、`GIN`、`GiST`、`BRIN`）。

#### **Day 7: 数据库设计基础**
- 学习数据库范式（第一、第二和第三范式）。
- 设计一个简单的关系型数据库模型（如图书管理系统）。

---

### **第 2 周：查询与高级操作**
#### **Day 8: 多表查询（JOIN）**
- 理解表之间的关系：一对一、一对多、多对多。
- 使用 `INNER JOIN`、`LEFT JOIN`、`RIGHT JOIN` 和 `FULL OUTER JOIN`。
- 实践常见的多表查询场景。

#### **Day 9: 聚合函数**
- 学习聚合函数：`COUNT`、`SUM`、`AVG`、`MIN`、`MAX`。
- 使用 `GROUP BY` 和 `HAVING` 子句。

#### **Day 10: 子查询**
- 简单子查询：`SELECT ... FROM (subquery)`。
- 相关子查询：`EXISTS` 和 `NOT EXISTS`。
- 使用子查询优化复杂查询。

#### **Day 11: 数据完整性**
- 学习主键（Primary Key）和外键（Foreign Key）。
- 设置唯一性约束（`UNIQUE`）。
- 了解检查约束（`CHECK`）。

#### **Day 12: 事务（Transaction）**
- 理解事务概念：`BEGIN`、`COMMIT` 和 `ROLLBACK`。
- 了解 ACID 属性。
- 使用 `SAVEPOINT` 实现部分回滚。

#### **Day 13: 视图（Views）**
- 创建、更新和删除视图：`CREATE VIEW`、`UPDATE VIEW`、`DROP VIEW`。
- 学习物化视图（`MATERIALIZED VIEW`）。

#### **Day 14: 数据导入与导出**
- 使用 `COPY` 命令导入和导出数据。
- 导入 CSV 文件和 JSON 文件。

---

### **第 3 周：性能优化与管理**
#### **Day 15: 查询性能优化**
- 使用 `EXPLAIN` 和 `EXPLAIN ANALYZE` 检查查询计划。
- 了解 PostgreSQL 的查询优化器。

#### **Day 16: 高级索引**
- 学习多列索引（Composite Index）。
- 创建部分索引和表达式索引。
- 了解全文搜索索引（`GIN` 和 `TSVECTOR`）。

#### **Day 17: 并发控制**
- 理解 PostgreSQL 的锁机制。
- 学习行级锁：`SELECT FOR UPDATE`。
- 解决死锁问题。

#### **Day 18: 数据库日志与监控**
- 配置和查看 PostgreSQL 日志。
- 使用 `pg_stat_activity` 监控活动查询。
- 学习 PostgreSQL 性能指标。

#### **Day 19: 分区表**
- 理解分区的用途。
- 创建和管理分区表：范围分区（Range Partitioning）和列表分区（List Partitioning）。

#### **Day 20: 数据库备份与恢复**
- 使用 `pg_dump` 和 `pg_restore`。
- 使用 `pg_basebackup` 进行物理备份。
- 实践备份和恢复策略。

---

### **第 4 周：高级功能与扩展**
#### **Day 21: JSON 与 NoSQL 功能**
- 存储和查询 JSON 数据。
- 使用 JSON 操作符（`->` 和 `->>`）。
- 学习 JSON 聚合和路径查询。

#### **Day 22: 用户与权限管理**
- 创建和管理用户：`CREATE USER` 和 `GRANT`。
- 设置角色权限。
- 实践多用户权限场景。

#### **Day 23: 触发器（Triggers）**
- 创建触发器：`CREATE TRIGGER`。
- 学习触发器函数（`PL/pgSQL`）。
- 实现触发器自动化任务。

#### **Day 24: 存储过程与函数**
- 创建存储函数：`CREATE FUNCTION`。
- 使用 PL/pgSQL 实现逻辑。
- 学习函数参数（`IN`、`OUT`、`INOUT`）。

#### **Day 25: 扩展（Extensions）**
- 安装和使用扩展：`CREATE EXTENSION`。
- 常用扩展：`pg_stat_statements`、`PostGIS`、`pgcrypto`。

#### **Day 26: 数据库集群**
- 理解数据库集群与分布式数据库。
- 学习 PostgreSQL 的多主集群（如 Citus）。

#### **Day 27: 高可用性与复制**
- 设置主从复制（Streaming Replication）。
- 学习逻辑复制（Logical Replication）。

---

### **最后 3 天：实践项目**
#### **Day 28: 项目设计**
- 设计一个多表数据库模型（如电商平台、学生管理系统）。

#### **Day 29: 项目实现**
- 实现 CRUD 功能、复杂查询和事务。
- 添加索引和优化性能。

#### **Day 30: 部署与总结**
- 将数据库部署到云服务（如 AWS RDS、Google Cloud SQL）。
- 总结学习成果，编写学习笔记。

---

### 工具与资源
- PostgreSQL 官方文档：[PostgreSQL 官方文档](https://www.postgresql.org/docs/)
- 在线练习平台：[DB Fiddle](https://www.db-fiddle.com/)
- 图形化工具：PgAdmin、DBeaver。

如果需要任何具体内容的代码或指导，请告诉我！