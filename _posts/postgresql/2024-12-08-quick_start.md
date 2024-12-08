---
layout: post
title:  安装 postgresql
date:   2024-12-07 11:24:29 +0800
categories: 
    - database
    - postgresql
---

### 连接数据库

切换到 PostgreSQL 管理员用户：

    sudo -i -u postgres

进入 psql：

    psql

    -U: 指定用户名。
    -d: 指定数据库名称。

### 基本命令

列出数据库：

    \l

切换数据库：

    \c database_name

列出表：

    \dt

查看帮助：

    \?

退出 psql：

    \q
<!-- 
设置 postgres 用户密码：

    ALTER USER postgres PASSWORD 'yourpassword';

退出 psql 并尝试重新连接：

    psql -U postgres -->