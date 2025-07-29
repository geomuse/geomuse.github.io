## ✅ 二、常见状态码详解（附 requests 示例）

### 📗 2xx 成功类

| 状态码   | 含义               | 示例说明               |
| ----- | ---------------- | ------------------ |
| `200` | OK，请求成功          | 最常见。成功获取网页、API 数据等 |
| `201` | Created，资源已创建    | 如 POST 创建资源成功      |
| `204` | No Content，无内容返回 | 通常用于 DELETE 操作     |

#### 示例代码：

```python
response = requests.get('https://httpbin.org/status/200')
if response.status_code == 200:
    print("请求成功")
```

---

### 📙 3xx 重定向类

| 状态码   | 含义               | 示例说明       |
| ----- | ---------------- | ---------- |
| `301` | 永久重定向            | 网址永久搬家     |
| `302` | 临时重定向            | 网址临时跳转     |
| `304` | Not Modified，未修改 | 浏览器缓存使用的响应 |

#### 自动跟随重定向：

```python
response = requests.get('http://httpbin.org/redirect/1')
print(response.url)  # 最终跳转后的URL
```

---

### 📕 4xx 客户端错误类

| 状态码   | 含义                     | 示例说明         |
| ----- | ---------------------- | ------------ |
| `400` | Bad Request，请求错误       | 参数格式不对、缺字段   |
| `401` | Unauthorized，未授权       | 需要登录或 Token  |
| `403` | Forbidden，被禁止访问        | 有权限问题        |
| `404` | Not Found，资源未找到        | URL 错误或数据不存在 |
| `429` | Too Many Requests，过载请求 | 爬虫被封、限流触发    |

#### 示例：

```python
response = requests.get('https://httpbin.org/status/404')
if response.status_code == 404:
    print("资源不存在")
```

---

### 📕 5xx 服务器错误类

| 状态码   | 含义                            | 示例说明     |
| ----- | ----------------------------- | -------- |
| `500` | Internal Server Error，服务器内部错误 | 程序崩溃或bug |
| `502` | Bad Gateway，网关错误              | 服务中间层出问题 |
| `503` | Service Unavailable，服务不可用     | 服务器过载或维护 |
| `504` | Gateway Timeout，网关超时          | 响应超时     |

---

## 🔍 三、如何用 requests 判断状态码

```python
response = requests.get('https://httpbin.org/get')

if response.ok:  # 等价于 200 <= code < 400
    print("请求成功")
elif response.status_code == 404:
    print("页面找不到")
else:
    print(f"错误代码：{response.status_code}")
```

---

## 📦 四、状态码速查表（开发中最常见）

| 状态码 | 含义            | 使用场景示例      |
| --- | ------------- | ----------- |
| 200 | 成功            | 正常请求 API 数据 |
| 201 | 创建成功          | 新增用户、文章等    |
| 204 | 删除成功，无返回      | 删除请求        |
| 400 | 请求有误          | 参数缺失、格式错误   |
| 401 | 未登录或 Token 失效 | 需要认证        |
| 403 | 权限不足          | 被封号、IP被限制   |
| 404 | 页面或数据不存在      | 地址打错、ID无效   |
| 429 | 请求过多          | 爬虫频率太快      |
| 500 | 服务器错误         | API出错       |
| 503 | 服务器维护中        | 需稍后重试       |

---

如果你需要我写一个自动处理状态码的模板函数，例如爬虫或API抓取时自动重试、限流保护，我也可以为你写出来。是否需要？
