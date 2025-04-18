---
layout: post
title : mongodb bson 与 json 差别
date : 2024-11-04 11:24:29 +0800
categories: 
    - mongodb
    - database
---

BSON（Binary JSON）和 JSON（JavaScript Object Notation）都是数据格式，但它们在结构、存储方式和使用场景上有一些关键的差异。

### BSON 与 JSON 的差别

1. **数据格式**：
   - **JSON** 是一种轻量级的文本数据格式，以纯文本表示，主要用于数据交换。它由键值对组成，支持简单的数据类型（如字符串、数值、布尔、数组和嵌套对象）。
   - **BSON** 是 JSON 的二进制格式，专为 MongoDB 设计，使用二进制编码来提高解析速度和效率。它比 JSON 更加紧凑，适合在网络传输中使用。

2. **支持的数据类型**：
   - **JSON** 的数据类型相对简单，只支持字符串、数字、布尔值、数组和对象。
   - **BSON** 除了 JSON 支持的数据类型，还支持更多类型（如 `Date`、`Binary Data`、`Decimal128`、`ObjectId` 等），这些类型对数据库操作更有用。

3. **数据大小**：
   - **JSON** 使用纯文本，所以通常占用的存储空间较大。
   - **BSON** 采用二进制编码，可以压缩数据，提高存储和传输效率。

4. **易读性**：
   - **JSON** 是可读的文本格式，便于人类直接理解和编辑。
   - **BSON** 是二进制格式，主要供机器使用，因此需要工具进行解析。

5. **性能**：
   - **JSON** 的解析速度较慢，因为它需要逐字符解析。
   - **BSON** 由于使用二进制存储，解析速度更快，尤其在处理大量数据时效果显著。

#### BSON 数据

在 MongoDB 中，BSON 是自动生成的。当你在 MongoDB 中插入一个 JSON 文档时，MongoDB 会自动将其转换为 BSON 格式进行存储。BSON 格式的文档可以使用 `bson` 库在 Node.js 或 Python 等环境中手动创建。

在 Python 中，可以使用 `pymongo` 库的 `bson` 模块生成 BSON 数据：

```python
from bson import BSON

person = {
    "name": "Alice",
    "age": 30,
    "city": "New York",
    "skills": ["JavaScript", "Python", "MongoDB"]
}

# 将字典转换为 BSON
bson_data = BSON.encode(person)
print(bson_data)
```

### 总结

- JSON 是易读的文本格式，适合小数据量传输和存储。
- BSON 是紧凑、高效的二进制格式，适合在 MongoDB 等数据库中存储和传输大数据量。