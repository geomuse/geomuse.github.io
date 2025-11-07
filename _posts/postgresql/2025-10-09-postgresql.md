---
layout: post
title:  数据库管理postgresql 查询数据通过pgadmin
date:   2025-10-09 09:01:00 +0800
image: 09.jpg
tags: 
    - python
    - postgresql
---

### 連接到您的 PostgreSQL 資料庫 

首先，確保您已經打開 **pgAdmin** 並連接到您想要創建此表的 **資料庫**。

-----

### 打開查詢工具 

有幾種方法可以打開查詢工具：

  * **右鍵點擊** 您選擇的資料庫，然後選擇 **“Query Tool”** (查詢工具)。
  * 或者，在左側的樹狀結構中選中您的資料庫後，點擊頂部工具欄上的 **“SQL”** 圖標（通常是一個帶有放大鏡的小圖標，位於“工具”菜單下）。

-----

### 執行 SQL 語句

將您提供的所有 SQL 語句一次性複製並貼上到查詢工具的編輯器中：

```sql
CREATE TABLE users(
    id SERIAL PRIMARY KEY, 
    name TEXT, 
    age INT
);

INSERT INTO users(name, age) 
VALUES ('Alice', 25);

SELECT * FROM users;
```

#### 執行步驟

1.  **複製** 上述程式碼並 **貼上** 到查詢編輯器中。
2.  點擊工具欄上的 **“Execute/Refresh”** (執行/重新整理) 按鈕（通常是一個**播放**▶️符號的圖標）。

-----

### 查看結果

執行後，您會看到以下三個操作的結果：

  * **`CREATE TABLE`** 和 **`INSERT`**：這兩個操作會在下方的 **"Messages"** (訊息) 或 **"Data Output"** (數據輸出) 標籤中顯示成功訊息 (例如: `CREATE TABLE` 和 `INSERT 0 1`)。
  * **`SELECT * FROM users`**：這個查詢的結果會顯示在 **"Data Output"** (數據輸出) 標籤中，您應該會看到如下結果：

| id | name | age |
| :--: | :----: | :-: |
| 1 | Alice | 25 |

-----

### 關於表結構

您創建的這個 `users` 表結構定義如下：

  * **`id SERIAL PRIMARY KEY`**: 是一個自動遞增的整數，用作表的唯一標識符。
  * **`name TEXT`**: 用於儲存字串（姓名）。
  * **`age INT`**: 用於儲存整數（年齡）。