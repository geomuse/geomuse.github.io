## **练习 2：阶乘计算（递归与循环）**  
**任务：**  
- 编写两个函数 `factorial_recursive(n)` 和 `factorial_loop(n)`，分别使用 **递归** 和 **循环** 计算 `n!`（n 的阶乘）。  
- `n! = n × (n-1) × (n-2) × ... × 1`，其中 `0! = 1`。  

**示例输入：**  
```python
print(factorial_recursive(5))
print(factorial_loop(5))
```
**示例输出：**  
```
120
120
```

---

## **练习 3：判断是否为回文数**  
**任务：**  
- 编写一个函数 `is_palindrome(n)`，判断一个整数是否是回文数（正着和反着读都一样）。  
- 例如：`121`、`1331` 是回文数，`123` 不是。  

**示例输入：**  
```python
print(is_palindrome(121))
print(is_palindrome(123))
```
**示例输出：**  
```
True
False
```

**提示：**  
- 可以转换为字符串 `str(n)`，然后 `[::-1]` 反转字符串进行判断。

---

## **练习 4：文件写入与读取**  
**任务：**  
- 让用户输入一个句子，并将其写入 `output.txt` 文件。  
- 之后，再读取 `output.txt`，并打印其中的内容。  

**示例输入：**  
```
请输入一句话：Hello, Python!
```
**示例输出（读取文件内容）：**  
```
文件内容：Hello, Python!
```

**提示：**  
- 使用 `open('output.txt', 'w', encoding='utf-8')` 写入文件。  
- 使用 `open('output.txt', 'r', encoding='utf-8')` 读取文件。

---

## **练习 5：统计文件中的单词数量**  
**任务：**  
- 读取 `sample.txt` 文件，并统计其中的单词数量。  

**示例内容（sample.txt）：**  
```
Python is a great programming language.
It is easy to learn and powerful.
```
**示例输出：**  
```
文件包含 12 个单词。
```

**提示：**  
- 使用 `split()` 方法按空格拆分单词。  
- 读取文件内容后 `words = content.split()`。

**Python 第三周练习（面向对象编程 & 模块）**  

目标：掌握 **类与对象、继承、多态、封装、模块与包**，加强对面向对象编程（OOP）的理解。  

---

## **练习 1：定义一个“学生”类**  
**任务：**  
- 创建一个 `Student` 类，包含以下属性和方法：
  - **属性**：`name`（姓名）、`age`（年龄）、`score`（成绩）  
  - **方法**：`display_info()`，打印学生信息  

**示例代码：**  
```python
s1 = Student("Alice", 20, 88)
s1.display_info()
```
**示例输出：**  
```
姓名：Alice, 年龄：20, 成绩：88
```

---

## **练习 2：银行账户类**  
**任务：**  
- 设计一个 `BankAccount` 类，具有以下功能：
  - **初始化**：账户余额 `balance`
  - **`deposit(amount)`** 方法：存款（增加余额）
  - **`withdraw(amount)`** 方法：取款（减少余额，余额不足时报错）
  - **`get_balance()`** 方法：返回当前余额  

**示例代码：**  
```python
account = BankAccount(1000)
account.deposit(500)
account.withdraw(200)
print(account.get_balance())
```
**示例输出：**  
```
1300
```

---

## **练习 3：动物继承**  
**任务：**  
- 定义一个 `Animal` 类，包含 `make_sound()` 方法（默认打印 `动物叫声`）。  
- 定义 `Dog` 和 `Cat` 继承 `Animal`，分别重写 `make_sound()`：
  - `Dog` 输出 `"汪汪！"`
  - `Cat` 输出 `"喵喵！"`

**示例代码：**  
```python
dog = Dog()
cat = Cat()
dog.make_sound()
cat.make_sound()
```
**示例输出：**  
```
汪汪！
喵喵！
```

---

## **练习 4：创建一个模块**  
**任务：**  
- 创建一个 `math_utils.py` 模块，包含以下函数：
  - `add(a, b)`: 返回两个数的和  
  - `subtract(a, b)`: 返回两个数的差  
  - `multiply(a, b)`: 返回两个数的积  
  - `divide(a, b)`: 返回两个数的商（如果 `b=0`，返回 `"除数不能为零"`）  
- 在 `main.py` 文件中导入 `math_utils` 并调用函数。

**示例代码（math_utils.py）：**  
```python
def add(a, b):
    return a + b
def subtract(a, b):
    return a - b
def multiply(a, b):
    return a * b
def divide(a, b):
    return "除数不能为零" if b == 0 else a / b
```
**示例代码（main.py）：**  
```python
import math_utils

print(math_utils.add(3, 5))
print(math_utils.divide(10, 2))
```
**示例输出：**  
```
8
5.0
```

---

## **练习 5：文件处理 & 学生管理系统**  
**任务：**  
- 设计一个简单的 **学生管理系统**，支持：
  - **添加学生信息**（姓名、年龄、成绩）
  - **保存到文件**（students.txt）
  - **从文件读取学生信息并显示**

**示例交互：**  
```
1. 添加学生
2. 显示所有学生
3. 退出
选择：1
请输入姓名：Alice
请输入年龄：20
请输入成绩：88
学生信息已保存！
```

**示例代码框架（补全代码）：**  
```python
class Student:
    def __init__(self, name, age, score):
        self.name = name
        self.age = age
        self.score = score

    def __str__(self):
        return f"{self.name}, {self.age}, {self.score}"

def save_student(student):
    with open("students.txt", "a") as file:
        file.write(str(student) + "\n")

def load_students():
    try:
        with open("students.txt", "r") as file:
            return file.readlines()
    except FileNotFoundError:
        return []

while True:
    print("1. 添加学生\n2. 显示所有学生\n3. 退出")
    choice = input("选择：")
    
    if choice == "1":
        name = input("请输入姓名：")
        age = input("请输入年龄：")
        score = input("请输入成绩：")
        s = Student(name, age, score)
        save_student(s)
        print("学生信息已保存！")
    elif choice == "2":
        students = load_students()
        for student in students:
            print(student.strip())
    elif choice == "3":
        break
    else:
        print("无效输入，请重试！")
```

---

这些练习涵盖了 **面向对象编程（OOP）、模块、文件处理**，尝试实现看看吧！如果需要讲解或完整代码，告诉我 😃

**Python 第四周练习（异常处理 & 进阶数据结构 & 多线程）**  

目标：掌握 **异常处理、字典 & 集合、列表推导式、多线程** 等 Python 进阶知识，提高编程能力。

---

## **练习 1：异常处理——计算器**  
**任务：**  
- 编写一个 **简单计算器**，支持加、减、乘、除。  
- **异常处理**：
  - 用户输入非法字符时，提示 `"请输入有效的数字"`  
  - 除数为 `0` 时，提示 `"除数不能为零"`  

**示例代码：**  
```python
def calculator():
    try:
        num1 = float(input("请输入第一个数字："))
        num2 = float(input("请输入第二个数字："))
        operation = input("请输入运算符 (+, -, *, /)：")

        if operation == "+":
            result = num1 + num2
        elif operation == "-":
            result = num1 - num2
        elif operation == "*":
            result = num1 * num2
        elif operation == "/":
            if num2 == 0:
                raise ZeroDivisionError("除数不能为零")
            result = num1 / num2
        else:
            raise ValueError("无效的运算符")

        print(f"计算结果：{result}")

    except ValueError as e:
        print(f"输入错误：{e}")
    except ZeroDivisionError as e:
        print(f"数学错误：{e}")

calculator()
```

---

## **练习 2：字典操作——词频统计**  
**任务：**  
- 统计一个英文句子中各单词出现的次数，并存入字典。  

**示例输入：**  
```python
sentence = "hello world hello Python"
```
**示例输出：**  
```python
{'hello': 2, 'world': 1, 'Python': 1}
```

**提示：**  
- 使用 `split()` 拆分单词  
- 遍历单词列表，使用 **字典** 存储单词和次数  

---

## **练习 3：列表推导式——生成平方数**  
**任务：**  
- 用 **列表推导式** 生成 1-10 的平方数列表。  

**示例代码：**  
```python
squares = [x ** 2 for x in range(1, 11)]
print(squares)
```
**示例输出：**  
```python
[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
```

---

## **练习 4：集合操作——去除重复单词**  
**任务：**  
- 让用户输入一个句子，去除重复单词，并按字母顺序排序。  

**示例输入：**  
```
hello world python python world
```
**示例输出：**  
```
hello python world
```

**提示：**  
- 使用 `set()` 去重  
- 使用 `sorted()` 排序  

---

## **练习 5：多线程——倒计时**  
**任务：**  
- 让程序 **倒计时 10 秒**，每秒打印当前秒数，并在倒计时结束后打印 `"时间到！"`  

**示例输出：**  
```
倒计时：10
倒计时：9
倒计时：8
...
时间到！
```

**提示：**  
- 使用 `time.sleep(1)` 让程序等待 1 秒  
- 使用 `threading.Thread` 启动倒计时  

**示例代码：**  
```python
import time
import threading

def countdown(n):
    while n > 0:
        print(f"倒计时：{n}")
        time.sleep(1)
        n -= 1
    print("时间到！")

thread = threading.Thread(target=countdown, args=(10,))
thread.start()
```

---

这些练习涉及 **异常处理、字典、集合、列表推导式、多线程**，是 Python 进阶必备技能，尝试实现看看吧！如果有问题，欢迎提问 😃

### **Python 第五周练习（正则表达式 & 网络请求 & 数据库）**  

**目标：**  
掌握 **正则表达式、网络爬取、SQLite 数据库操作**，强化数据处理和存储能力。  

---

## **练习 1：正则表达式——匹配邮箱**  
**任务：**  
- 让用户输入一个字符串，检查其中是否包含有效的 **邮箱地址**（格式：`xxx@xxx.com`）。  
- 使用 **正则表达式（re 模块）** 进行匹配。  

**示例输入：**  
```
请输入邮箱：test.email@domain.com
```
**示例输出：**  
```
有效邮箱！
```

**提示：**  
- 使用 `re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", email)` 进行匹配。  

---

## **练习 2：正则表达式——提取手机号**  
**任务：**  
- 让用户输入一段文本，提取其中的 **手机号**（假设手机号格式为 **11 位数字**）。  
- 使用 **正则表达式** 进行查找。  

**示例输入：**  
```
用户输入：我的手机号是 13812345678，还有个备用号码 18998765432。
```
**示例输出：**  
```
提取的手机号：['13812345678', '18998765432']
```

**提示：**  
- 使用 `re.findall(r"\b1[3-9]\d{9}\b", text)` 进行匹配。  

---

## **练习 3：网络请求——获取网页内容**  
**任务：**  
- 使用 `requests` 库获取 **百度首页** 的 HTML 代码，并显示前 500 个字符。  

**示例代码：**  
```python
import requests

url = "https://www.baidu.com"
response = requests.get(url)
print(response.text[:500])  # 仅显示前 500 个字符
```

**输出示例（部分）：**  
```
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>百度一下，你就知道</title>
...
```

**提示：**  
- `requests.get(url).text` 获取网页源码  
- 可能需要 `pip install requests` 安装 `requests` 库  

---

## **练习 4：爬取 JSON 数据**  
**任务：**  
- 获取 **随机用户 API** 的数据，并解析其中的 `name` 和 `email`。  

**API 地址：**  
```
https://randomuser.me/api/
```

**示例代码：**  
```python
import requests

url = "https://randomuser.me/api/"
response = requests.get(url).json()

name = response["results"][0]["name"]
email = response["results"][0]["email"]

print(f"姓名：{name['first']} {name['last']}")
print(f"邮箱：{email}")
```

**示例输出：**  
```
姓名：John Doe
邮箱：johndoe@example.com
```

---

## **练习 5：SQLite 数据库——存储用户信息**  
**任务：**  
- **创建 SQLite 数据库** `users.db`，并建立 `users` 表（包含 `id, name, age, email`）。  
- **插入数据**（例如 `"Alice", 25, "alice@example.com"`）。  
- **查询所有数据** 并打印出来。  

**示例代码：**  
```python
import sqlite3

# 连接数据库（如果不存在则创建）
conn = sqlite3.connect("users.db")
cursor = conn.cursor()

# 创建表
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age INTEGER,
    email TEXT
)
""")

# 插入数据
cursor.execute("INSERT INTO users (name, age, email) VALUES (?, ?, ?)", 
               ("Alice", 25, "alice@example.com"))

# 提交事务
conn.commit()

# 查询数据
cursor.execute("SELECT * FROM users")
rows = cursor.fetchall()

# 打印查询结果
for row in rows:
    print(row)

# 关闭连接
conn.close()
```

**示例输出：**  
```
(1, 'Alice', 25, 'alice@example.com')
```

**提示：**  
- `sqlite3.connect("users.db")` 连接数据库  
- `cursor.execute()` 执行 SQL 语句  
- `fetchall()` 获取所有查询结果  

---

### **总结**
本周练习涵盖 **正则表达式、网络请求、数据库操作**，是 Python 数据处理的核心技能。尝试练习看看吧！如果有问题，欢迎提问 😃

### **Python 第六周练习（数据分析 & 多进程 & GUI 编程）**  

**目标：**  
掌握 **Pandas 数据分析、Matplotlib 可视化、多进程处理、Tkinter GUI**，提升数据处理与应用开发能力。  

---

## **练习 1：Pandas 读取 CSV 文件**  
**任务：**  
- 使用 **Pandas** 读取 `data.csv` 文件（格式如下）：  
  ```
  name,age,score
  Alice,25,88
  Bob,22,76
  Charlie,23,90
  ```
- 计算 **平均成绩**，并输出所有 `score` **大于 80** 的学生信息。  

**示例代码：**  
```python
import pandas as pd

# 读取 CSV 文件
df = pd.read_csv("data.csv")

# 计算平均成绩
avg_score = df["score"].mean()
print(f"平均成绩：{avg_score}")

# 筛选成绩大于 80 的学生
high_scorers = df[df["score"] > 80]
print("成绩大于 80 的学生：")
print(high_scorers)
```

**示例输出：**  
```
平均成绩：84.67
成绩大于 80 的学生：
     name  age  score
0  Alice   25     88
2  Charlie  23     90
```

---

## **练习 2：Matplotlib 绘制折线图**  
**任务：**  
- 用 `Matplotlib` **绘制成绩趋势图**，显示学生的成绩变化趋势。  

**示例代码：**  
```python
import matplotlib.pyplot as plt

# 示例数据
students = ["Alice", "Bob", "Charlie"]
scores = [88, 76, 90]

# 绘制折线图
plt.plot(students, scores, marker="o", linestyle="-", color="b")
plt.xlabel("学生")
plt.ylabel("成绩")
plt.title("学生成绩变化趋势")
plt.show()
```

**示例输出：**  
📊（折线图展示学生成绩）

---

## **练习 3：多进程计算——计算 1~1000000 之和**  
**任务：**  
- 使用 **多进程** 计算 **1~1000000 之和**，提升计算速度。  
- 采用 `multiprocessing` 将任务拆分为 **4 个进程** 进行计算。  

**示例代码：**  
```python
import multiprocessing

def sum_range(start, end, result, index):
    result[index] = sum(range(start, end))

if __name__ == "__main__":
    num_processes = 4
    n = 1000000
    step = n // num_processes

    manager = multiprocessing.Manager()
    result = manager.dict()

    processes = []
    for i in range(num_processes):
        start = i * step + 1
        end = (i + 1) * step + 1 if i != num_processes - 1 else n + 1
        p = multiprocessing.Process(target=sum_range, args=(start, end, result, i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    total_sum = sum(result.values())
    print(f"1 到 {n} 的总和：{total_sum}")
```

**示例输出：**  
```
1 到 1000000 的总和：500000500000
```

---

## **练习 4：Tkinter GUI 计算器**  
**任务：**  
- 使用 **Tkinter** 实现一个 **简单计算器**，支持 **加、减、乘、除** 运算。  

**示例界面：**  
📱（GUI 界面，包含数字输入框、运算符按钮和计算结果显示）

**示例代码（基础框架）：**  
```python
import tkinter as tk

def calculate():
    try:
        expr = entry.get()
        result.set(eval(expr))
    except:
        result.set("错误")

# 创建窗口
root = tk.Tk()
root.title("计算器")

# 输入框
entry = tk.Entry(root, width=20)
entry.grid(row=0, column=0, columnspan=4)

# 结果显示
result = tk.StringVar()
label = tk.Label(root, textvariable=result, width=20, height=2)
label.grid(row=1, column=0, columnspan=4)

# 按钮
buttons = [
    ("7", 2, 0), ("8", 2, 1), ("9", 2, 2), ("/", 2, 3),
    ("4", 3, 0), ("5", 3, 1), ("6", 3, 2), ("*", 3, 3),
    ("1", 4, 0), ("2", 4, 1), ("3", 4, 2), ("-", 4, 3),
    ("0", 5, 0), (".", 5, 1), ("=", 5, 2), ("+", 5, 3),
]

for (text, row, col) in buttons:
    action = lambda x=text: entry.insert(tk.END, x) if x != "=" else calculate()
    tk.Button(root, text=text, width=5, command=action).grid(row=row, column=col)

# 运行主循环
root.mainloop()
```

**功能：**  
- 用户输入 `5+3*2`，点击 `=`，计算器显示 `11`。  

---

这些练习涵盖 **数据分析、可视化、多进程、GUI**，可以大大提高 Python 实战能力！🚀

### **Python 第七周练习（面向对象 & 文件操作 & API 开发）**  

**目标：**  
掌握 **面向对象编程（OOP）、文件操作、Flask API 开发**，提升 Python 实战能力。  

---

## **练习 1：面向对象编程（OOP）——银行账户管理**  
**任务：**  
- 定义一个 **BankAccount** 类，包含：  
  - `__init__(self, owner, balance=0)`: 账户所有者和初始余额  
  - `deposit(self, amount)`: 存款方法  
  - `withdraw(self, amount)`: 取款方法（余额不足时报错）  
  - `get_balance(self)`: 查询余额  

**示例代码：**  
```python
class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount
        print(f"{self.owner} 存入 {amount} 元，当前余额：{self.balance}")

    def withdraw(self, amount):
        if amount > self.balance:
            print("余额不足，无法取款")
        else:
            self.balance -= amount
            print(f"{self.owner} 取出 {amount} 元，当前余额：{self.balance}")

    def get_balance(self):
        return self.balance

# 测试
account = BankAccount("Alice", 1000)
account.deposit(500)
account.withdraw(300)
account.withdraw(1500)  # 余额不足
```

**示例输出：**  
```
Alice 存入 500 元，当前余额：1500
Alice 取出 300 元，当前余额：1200
余额不足，无法取款
```

---

## **练习 2：文件操作——日志记录**  
**任务：**  
- **创建 log.txt 文件**，每次运行程序时，向文件中追加一条日志（包含当前时间和消息）。  

**示例代码：**  
```python
import datetime

def write_log(message):
    with open("log.txt", "a") as file:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"[{timestamp}] {message}\n")

# 记录日志
write_log("程序启动")
write_log("执行某个操作")
write_log("程序结束")
```

**示例 `log.txt` 文件内容：**  
```
[2025-02-24 10:00:00] 程序启动
[2025-02-24 10:05:00] 执行某个操作
[2025-02-24 10:10:00] 程序结束
```

---

## **练习 3：JSON 文件存储——用户信息管理**  
**任务：**  
- 让用户输入 **姓名、年龄、邮箱**，存入 `users.json` 文件。  
- 读取 `users.json` 文件并显示所有用户信息。  

**示例代码：**  
```python
import json

def save_user(name, age, email):
    try:
        with open("users.json", "r") as file:
            users = json.load(file)
    except FileNotFoundError:
        users = []

    users.append({"name": name, "age": age, "email": email})

    with open("users.json", "w") as file:
        json.dump(users, file, indent=4)

def load_users():
    try:
        with open("users.json", "r") as file:
            users = json.load(file)
            for user in users:
                print(user)
    except FileNotFoundError:
        print("没有找到用户数据")

# 测试
save_user("Alice", 25, "alice@example.com")
save_user("Bob", 30, "bob@example.com")
load_users()
```

**示例 `users.json` 文件内容：**  
```json
[
    {"name": "Alice", "age": 25, "email": "alice@example.com"},
    {"name": "Bob", "age": 30, "email": "bob@example.com"}
]
```

---

## **练习 4：Flask API——简单的 Web 服务**  
**任务：**  
- 使用 `Flask` 创建一个 **简单 API**，包含：
  - `/`：返回 `"欢迎使用 API"`
  - `/user/<name>`：返回 `{"message": "你好, name!"}`  

**示例代码：**  
```python
from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "欢迎使用 API"

@app.route("/user/<name>")
def user(name):
    return jsonify({"message": f"你好, {name}!"})

if __name__ == "__main__":
    app.run(debug=True)
```

**运行方式：**  
- 运行 `python app.py`  
- 在浏览器访问 `http://127.0.0.1:5000/`  
- 访问 `http://127.0.0.1:5000/user/Alice` 得到：
  ```json
  {"message": "你好, Alice!"}
  ```

---

### **第七周练习总结**
✅ **掌握 OOP（类与对象）**  
✅ **熟练文件读写（txt、JSON）**  
✅ **学习 Flask API 开发**  

可以尝试修改代码，添加新功能，提升实战能力！🚀

### **Python 第八周练习（Flask Web 开发 & WebSocket & 线程编程）**  

**目标：**  
掌握 **Flask Web 框架、WebSocket 实时通信、Python 线程编程**，提升 Python 的 Web 应用能力。  

---

## **练习 1：Flask Web API 开发**  
**任务：**  
- 使用 `Flask` 创建一个 **简单 API**，支持以下功能：  
  - **GET /users** 获取所有用户  
  - **POST /users** 添加新用户  

**示例代码：**  
```python
from flask import Flask, request, jsonify

app = Flask(__name__)

# 用户数据存储
users = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]

# 获取所有用户
@app.route("/users", methods=["GET"])
def get_users():
    return jsonify(users)

# 添加用户
@app.route("/users", methods=["POST"])
def add_user():
    data = request.json
    new_user = {"id": len(users) + 1, "name": data["name"]}
    users.append(new_user)
    return jsonify(new_user), 201

if __name__ == "__main__":
    app.run(debug=True)
```

**测试方法（运行 Flask 服务器后）：**  
- **获取用户列表**（在浏览器访问）：  
  ```
  http://127.0.0.1:5000/users
  ```
- **添加用户**（使用 `Postman` 发送 `POST` 请求）：  
  ```json
  {"name": "Charlie"}
  ```

**示例输出：**  
```
[
    {"id": 1, "name": "Alice"},
    {"id": 2, "name": "Bob"},
    {"id": 3, "name": "Charlie"}
]
```

---

## **练习 2：WebSocket 实时聊天**  
**任务：**  
- 使用 `Flask-SocketIO` 搭建 **WebSocket 服务器**，实现简单 **聊天室功能**。  

**示例代码：**  
```python
from flask import Flask, render_template
from flask_socketio import SocketIO, send

app = Flask(__name__)
socketio = SocketIO(app)

# 处理客户端消息
@socketio.on("message")
def handle_message(msg):
    print(f"收到消息: {msg}")
    send(msg, broadcast=True)

if __name__ == "__main__":
    socketio.run(app, debug=True)
```

**测试步骤：**  
1. **安装 Flask-SocketIO**  
   ```bash
   pip install flask-socketio
   ```
2. **运行服务器**  
   ```bash
   python app.py
   ```
3. **使用 WebSocket 客户端（如 `wscat`）连接服务器**  
   ```bash
   wscat -c ws://127.0.0.1:5000/
   ```

---

## **练习 3：多线程并发任务**  
**任务：**  
- 创建 **两个线程**，一个 **打印奇数**，另一个 **打印偶数**，交替执行。  

**示例代码：**  
```python
import threading
import time

def print_odd():
    for i in range(1, 20, 2):
        print(f"奇数：{i}")
        time.sleep(0.5)

def print_even():
    for i in range(2, 21, 2):
        print(f"偶数：{i}")
        time.sleep(0.5)

# 创建线程
t1 = threading.Thread(target=print_odd)
t2 = threading.Thread(target=print_even)

# 启动线程
t1.start()
t2.start()

# 等待线程结束
t1.join()
t2.join()
```

**示例输出（交替出现）：**  
```
奇数：1
偶数：2
奇数：3
偶数：4
...
```

---

这周的练习主要围绕 **Flask Web API、WebSocket 实时通信、多线程编程**，如果有不清楚的地方可以随时问我！🚀

### **Python 第九周练习（数据库 & Web 爬虫 & 数据可视化）**  

**目标：**  
掌握 **SQLite 数据库、BeautifulSoup 爬虫、Matplotlib 进阶可视化**，提升 Python 在数据处理和应用开发中的能力。  

---

## **练习 1：SQLite 数据库操作**  
**任务：**  
- 使用 **SQLite** 创建 `users` 数据表，包含 `id, name, age` 三个字段。  
- 插入 **多个用户数据**，并查询 **所有用户信息**。  

**示例代码：**  
```python
import sqlite3

# 连接数据库（不存在则创建）
conn = sqlite3.connect("test.db")
cursor = conn.cursor()

# 创建 users 表
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    age INTEGER NOT NULL
)
""")

# 插入数据
cursor.execute("INSERT INTO users (name, age) VALUES ('Alice', 25)")
cursor.execute("INSERT INTO users (name, age) VALUES ('Bob', 30)")
conn.commit()  # 提交事务

# 查询数据
cursor.execute("SELECT * FROM users")
users = cursor.fetchall()
for user in users:
    print(user)

# 关闭连接
conn.close()
```

**示例输出：**  
```
(1, 'Alice', 25)
(2, 'Bob', 30)
```

---

## **练习 2：BeautifulSoup 爬虫获取网页标题**  
**任务：**  
- 爬取 **指定网页**（例如 `https://quotes.toscrape.com/`），提取 **页面标题** 和 **所有名人名言**。  

**示例代码：**  
```python
import requests
from bs4 import BeautifulSoup

# 目标网址
url = "https://quotes.toscrape.com/"
response = requests.get(url)

# 解析 HTML
soup = BeautifulSoup(response.text, "html.parser")

# 提取标题
title = soup.title.text
print(f"页面标题：{title}")

# 提取所有名言
quotes = soup.find_all("span", class_="text")
for quote in quotes:
    print(quote.text)
```

**示例输出：**  
```
页面标题：Quotes to Scrape
“The greatest glory in living lies not in never falling, but in rising every time we fall.”
...
```

---

## **练习 3：Matplotlib 高级可视化（柱状图 & 饼图）**  
**任务：**  
- 使用 `Matplotlib` 绘制 **年龄分布柱状图** 和 **性别比例饼图**。  

**示例代码（柱状图）：**  
```python
import matplotlib.pyplot as plt

# 示例数据
names = ["Alice", "Bob", "Charlie", "David"]
ages = [25, 30, 35, 40]

plt.bar(names, ages, color=["blue", "green", "red", "purple"])
plt.xlabel("姓名")
plt.ylabel("年龄")
plt.title("用户年龄分布")
plt.show()
```

**示例代码（饼图）：**  
```python
labels = ["男性", "女性"]
sizes = [60, 40]  # 男性60%，女性40%

plt.pie(sizes, labels=labels, autopct="%1.1f%%", colors=["blue", "pink"])
plt.title("性别比例")
plt.show()
```

**示例输出：**  
📊（柱状图 & 饼图显示年龄分布和性别比例）

---

这一周主要练习了 **SQLite 数据库操作、网页爬取和数据可视化**，如果有问题欢迎提问！🚀

### **Python 第十周练习（多进程 & 机器学习入门 & 自动化脚本）**  

**目标：**  
掌握 **Python 多进程编程、Scikit-Learn 机器学习基础、Selenium 自动化脚本**，提升 Python 在高性能计算、人工智能和自动化领域的应用能力。  

---

## **练习 1：多进程并行计算**  
**任务：**  
- 使用 `multiprocessing` **并行计算** **1~1000000** 的平方和，加速计算。  

**示例代码：**  
```python
import multiprocessing

def square_sum(start, end):
    """计算 start 到 end 之间所有数的平方和"""
    return sum(i * i for i in range(start, end))

if __name__ == "__main__":
    cpu_count = multiprocessing.cpu_count()
    print(f"CPU 核心数：{cpu_count}")

    # 任务拆分
    n = 1000000
    step = n // cpu_count
    processes = []
    results = []

    with multiprocessing.Pool(cpu_count) as pool:
        for i in range(cpu_count):
            start = i * step
            end = (i + 1) * step if i < cpu_count - 1 else n
            results.append(pool.apply_async(square_sum, (start, end)))

        total_sum = sum(res.get() for res in results)
    
    print(f"平方和结果：{total_sum}")
```

**要点：**  
✅ **使用 `multiprocessing.Pool` 并行计算**  
✅ **提高大规模数据计算效率**  

---

## **练习 2：机器学习入门（线性回归预测房价）**  
**任务：**  
- 使用 `scikit-learn` 训练 **线性回归模型**，预测房价。  

**示例代码：**  
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 训练数据（面积 & 房价）
X = np.array([50, 60, 70, 80, 90, 100]).reshape(-1, 1)  # 平方米
y = np.array([200, 240, 280, 320, 360, 400])  # 房价（万元）

# 训练模型
model = LinearRegression()
model.fit(X, y)

# 预测新房价
new_size = np.array([[110]])
predicted_price = model.predict(new_size)
print(f"预测 110 平方米房价：{predicted_price[0]:.2f} 万元")

# 绘制回归曲线
plt.scatter(X, y, color="blue", label="真实数据")
plt.plot(X, model.predict(X), color="red", label="回归线")
plt.xlabel("面积 (平方米)")
plt.ylabel("房价 (万元)")
plt.legend()
plt.show()
```

**要点：**  
✅ **使用 `LinearRegression()` 训练模型**  
✅ **绘制回归曲线，观察预测结果**  

---

## **练习 3：Selenium 自动化网页操作**  
**任务：**  
- 使用 `Selenium` 自动化 **打开 Google 并搜索“Python”**。  

**示例代码（需安装 ChromeDriver）：**  
```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

# 启动浏览器
driver = webdriver.Chrome()

# 访问 Google
driver.get("https://www.google.com")

# 搜索框输入 "Python"
search_box = driver.find_element(By.NAME, "q")
search_box.send_keys("Python")
search_box.send_keys(Keys.RETURN)

# 等待结果加载
time.sleep(3)

# 截图保存
driver.save_screenshot("google_search.png")

# 关闭浏览器
driver.quit()
```

**要点：**  
✅ **Selenium 远程控制浏览器**  
✅ **模拟人类输入 & 搜索**  
✅ **自动截图保存搜索结果**  

---

**总结：**  
这周练习了 **多进程计算、机器学习基础、自动化网页操作**，如果有问题欢迎提问！🚀

### **Python 第十一周练习（FastAPI 后端开发 & 并发编程 & OpenCV 处理图像）**  

**目标：**  
掌握 **FastAPI 构建高性能 API、asyncio 并发编程、OpenCV 图像处理**，提升 Python 在后端开发、异步编程和计算机视觉领域的能力。  

---

## **练习 1：FastAPI 构建 RESTful API**  
**任务：**  
- 使用 `FastAPI` **创建一个简单 API**，包含以下功能：  
  - **GET /items** 获取所有物品  
  - **POST /items** 添加新物品  

**示例代码：**  
```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float

items = []

@app.get("/items", response_model=List[Item])
def get_items():
    return items

@app.post("/items", response_model=Item)
def add_item(item: Item):
    items.append(item)
    return item
```

**运行 FastAPI 服务器**  
```bash
uvicorn filename:app --reload
```

**测试 API：**  
1. **获取物品列表**（浏览器访问）：  
   ```
   http://127.0.0.1:8000/items
   ```
2. **添加物品**（用 `Postman` 发送 `POST` 请求）：  
   ```json
   {"name": "Laptop", "price": 999.99}
   ```

**要点：**  
✅ **FastAPI 提供超快的 API 开发体验**  
✅ **自动生成 Swagger 文档**（访问 `http://127.0.0.1:8000/docs`）  

---

## **练习 2：asyncio 并发任务处理**  
**任务：**  
- 使用 `asyncio` **异步运行多个任务**，模拟 **爬取 3 个网站数据**。  

**示例代码：**  
```python
import asyncio

async def fetch_data(site):
    print(f"开始爬取 {site} ...")
    await asyncio.sleep(2)  # 模拟网络请求
    print(f"完成爬取 {site}！")

async def main():
    sites = ["https://site1.com", "https://site2.com", "https://site3.com"]
    tasks = [fetch_data(site) for site in sites]
    await asyncio.gather(*tasks)

asyncio.run(main())
```

**示例输出：**  
```
开始爬取 https://site1.com ...
开始爬取 https://site2.com ...
开始爬取 https://site3.com ...
完成爬取 https://site1.com！
完成爬取 https://site2.com！
完成爬取 https://site3.com！
```

**要点：**  
✅ **使用 `asyncio.gather()` 并行执行多个异步任务**  
✅ **提高网络 IO 任务的执行效率**  

---

## **练习 3：OpenCV 处理图像**  
**任务：**  
- 使用 `OpenCV` **读取 & 显示图像**，并 **转换为灰度图像**。  

**示例代码：**  
```python
import cv2

# 读取图片
image = cv2.imread("image.jpg")

# 显示原图
cv2.imshow("Original", image)

# 转换为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Image", gray)

# 等待按键后关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
```

**要点：**  
✅ **使用 `cv2.imread()` 读取图片**  
✅ **使用 `cv2.cvtColor()` 进行颜色转换**  
✅ **使用 `cv2.imshow()` 显示图像**  

---

**总结：**  
这周练习了 **FastAPI 构建 API、asyncio 并发编程、OpenCV 进行图像处理**，如果有问题欢迎提问！🚀

### **Python 第十二周练习（数据分析 & 深度学习入门 & 网络安全基础）**  

**目标：**  
掌握 **Pandas 进行数据分析、PyTorch 训练深度学习模型、Kali Linux 进行基础网络安全测试**，提升 Python 在数据科学、人工智能和网络安全领域的能力。  

---

## **练习 1：Pandas 进行数据分析**  
**任务：**  
- 读取 `CSV` 文件，分析数据集，计算 **平均值、最大值、最小值**。  

**示例代码：**  
```python
import pandas as pd

# 读取数据（假设有一个 data.csv 文件）
df = pd.read_csv("data.csv")

# 显示数据前 5 行
print(df.head())

# 计算统计信息
print("平均值：\n", df.mean())
print("最大值：\n", df.max())
print("最小值：\n", df.min())
```

**示例输出：**  
```
   name  age  salary
0  Alice   25   5000
1    Bob   30   7000
2  Carol   27   6500
...
平均值：
 age       27.3
salary   6166.7
dtype: float64
```

**要点：**  
✅ **使用 `pd.read_csv()` 读取 CSV 文件**  
✅ **使用 `df.mean(), df.max(), df.min()` 进行数据分析**  

---

## **练习 2：PyTorch 训练简单神经网络**  
**任务：**  
- 训练一个 **简单的神经网络**，实现 **二分类任务（手写数字识别）**。  

**示例代码：**  
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 数据加载
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 创建 DataLoader
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

# 定义神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(28*28, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.fc(x)

# 训练模型
model = SimpleNN()
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(3):  # 训练 3 轮
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

print("训练完成！")
```

**要点：**  
✅ **使用 `torchvision.datasets.MNIST` 读取手写数字数据**  
✅ **定义 `SimpleNN` 进行分类任务**  
✅ **使用 `CrossEntropyLoss()` 计算损失**  

---

## **练习 3：Kali Linux 进行基础网络安全测试（Python 脚本）**  
**任务：**  
- **使用 Python 进行端口扫描**，检测目标主机开放的端口。  

**示例代码：**  
```python
import socket

def scan_ports(target_ip, ports):
    for port in ports:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((target_ip, port))
        if result == 0:
            print(f"端口 {port} 开放")
        sock.close()

# 目标 IP 和端口列表
target = "192.168.1.1"
ports = [21, 22, 80, 443, 8080]

scan_ports(target, ports)
```

**示例输出（如果 80 端口开放）：**  
```
端口 80 开放
```

**要点：**  
✅ **使用 `socket` 进行端口扫描**  
✅ **扫描目标服务器的开放端口**  

---

**总结：**  
这周练习了 **Pandas 进行数据分析、PyTorch 训练神经网络、Python 进行端口扫描**，如果有问题欢迎提问！🚀

### **Python 第十三周练习（数据可视化 & Flask Web 开发 & 线程池优化）**  

**目标：**  
掌握 **Matplotlib & Seaborn 进行数据可视化、Flask 开发 Web 应用、多线程任务优化**，提升 Python 在数据展示、Web 后端开发和并发计算领域的能力。  

---

## **练习 1：Matplotlib & Seaborn 数据可视化**  
**任务：**  
- **绘制柱状图和折线图**，展示产品销量变化趋势。  

**示例代码（柱状图 + 折线图）：**  
```python
import matplotlib.pyplot as plt
import seaborn as sns

# 数据
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
sales = [120, 150, 180, 220, 260, 300]

# 设置 Seaborn 样式
sns.set_style("whitegrid")

# 绘制柱状图
plt.figure(figsize=(8, 5))
plt.bar(months, sales, color="blue", alpha=0.7, label="Monthly Sales")
plt.plot(months, sales, color="red", marker="o", linestyle="--", label="Trend")

plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Monthly Sales Trend")
plt.legend()
plt.show()
```

**要点：**  
✅ **`plt.bar()` 生成柱状图**  
✅ **`plt.plot()` 叠加折线图**  
✅ **`sns.set_style("whitegrid")` 美化可视化效果**  

---

## **练习 2：Flask 开发 Web 应用**  
**任务：**  
- **使用 Flask 构建 Web 服务器**，创建首页 **`/`** 和 **用户信息 API `GET /user/<name>`**。  

**示例代码：**  
```python
from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "欢迎来到 Flask Web 应用！"

@app.route("/user/<name>")
def user(name):
    return jsonify({"user": name, "message": f"Hello, {name}!"})

if __name__ == "__main__":
    app.run(debug=True)
```

**运行 Flask 服务器：**  
```bash
python app.py
```

**测试 API：**  
1. **访问首页**：  
   ```
   http://127.0.0.1:5000/
   ```
2. **访问用户 API**（输入 `http://127.0.0.1:5000/user/Alice`）：  
   ```json
   {"user": "Alice", "message": "Hello, Alice!"}
   ```

**要点：**  
✅ **使用 `Flask` 搭建 Web 服务器**  
✅ **返回 JSON 响应**  

---

## **练习 3：使用 ThreadPoolExecutor 进行多线程任务优化**  
**任务：**  
- **使用 `ThreadPoolExecutor` 并行执行多个任务**，提高程序效率。  

**示例代码（并发任务执行）：**  
```python
import concurrent.futures
import time

def worker(task_id):
    print(f"任务 {task_id} 开始...")
    time.sleep(2)  # 模拟耗时任务
    print(f"任务 {task_id} 完成！")
    return f"任务 {task_id} 结果"

# 创建线程池
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(worker, i) for i in range(5)]
    
    for future in concurrent.futures.as_completed(futures):
        print(future.result())
```

**示例输出：**  
```
任务 0 开始...
任务 1 开始...
任务 2 开始...
任务 0 完成！
任务 任务 1 完成！
2 完成！
任务 3 开始...
任务 4 开始...
任务 3 完成！
任务 4 完成！
```

**要点：**  
✅ **使用 `ThreadPoolExecutor` 并行执行任务**  
✅ **提高 I/O 任务执行效率（如爬虫、文件处理）**  

---

**总结：**  
这周练习了 **数据可视化、Flask Web 开发、线程池并发优化**，如果有问题欢迎提问！🚀

### **Python 第十四周练习（数据库操作 & Scrapy 爬虫 & 进程池优化）**  

**目标：**  
掌握 **SQLite 数据库操作、Scrapy 爬取网页数据、multiprocessing 进行进程池优化**，提升 Python 在 **数据库管理、数据采集、多进程计算** 方面的能力。  

---

## **练习 1：SQLite 数据库操作**  
**任务：**  
- **创建 SQLite 数据库**，插入用户数据，并执行查询操作。  

**示例代码（SQLite 增删查改）：**  
```python
import sqlite3

# 连接数据库（如果不存在则创建）
conn = sqlite3.connect("users.db")
cursor = conn.cursor()

# 创建用户表
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age INTEGER
)
""")

# 插入数据
cursor.execute("INSERT INTO users (name, age) VALUES (?, ?)", ("Alice", 25))
cursor.execute("INSERT INTO users (name, age) VALUES (?, ?)", ("Bob", 30))

# 查询数据
cursor.execute("SELECT * FROM users")
rows = cursor.fetchall()
for row in rows:
    print(row)

# 关闭数据库
conn.commit()
conn.close()
```

**示例输出：**  
```
(1, 'Alice', 25)
(2, 'Bob', 30)
```

**要点：**  
✅ **使用 `sqlite3.connect()` 连接数据库**  
✅ **`execute()` 执行 SQL 语句**  
✅ **使用 `fetchall()` 查询数据**  

---

## **练习 2：使用 Scrapy 爬取网页数据**  
**任务：**  
- **爬取 Quotes 网站（http://quotes.toscrape.com）**，获取名人名言。  

**步骤：**  
1. **安装 Scrapy**  
   ```bash
   pip install scrapy
   ```
2. **创建 Scrapy 项目**  
   ```bash
   scrapy startproject quotes_spider
   cd quotes_spider
   ```
3. **编写 Scrapy 爬虫（在 `spiders` 目录下创建 `quotes.py`）**  

**示例代码（`quotes.py`）：**  
```python
import scrapy

class QuotesSpider(scrapy.Spider):
    name = "quotes"
    start_urls = ["http://quotes.toscrape.com"]

    def parse(self, response):
        for quote in response.css("div.quote"):
            yield {
                "text": quote.css("span.text::text").get(),
                "author": quote.css("small.author::text").get(),
            }
```

4. **运行爬虫**  
   ```bash
   scrapy crawl quotes -o quotes.json
   ```

**示例输出（quotes.json）：**  
```json
[
    {"text": "“The world as we have created it is a process of our thinking.”", "author": "Albert Einstein"},
    {"text": "“It is our choices that show what we truly are...”", "author": "J.K. Rowling"}
]
```

**要点：**  
✅ **使用 Scrapy 爬取网页数据**  
✅ **提取 `quote.css("span.text::text").get()` 进行数据解析**  
✅ **保存数据到 JSON 文件**  

---

## **练习 3：使用 multiprocessing 进行进程池优化**  
**任务：**  
- **使用 `multiprocessing.Pool` 并行执行多个计算任务**，提高计算速度。  

**示例代码（计算平方并行处理）：**  
```python
import multiprocessing
import time

def square(n):
    print(f"计算 {n} 的平方...")
    time.sleep(1)
    return n * n

if __name__ == "__main__":
    numbers = [1, 2, 3, 4, 5]

    # 创建进程池
    with multiprocessing.Pool(processes=3) as pool:
        results = pool.map(square, numbers)

    print("计算结果:", results)
```

**示例输出（多个进程并行计算）：**  
```
计算 1 的平方...
计算 2 的平方...
计算 3 的平方...
计算 4 的平方...
计算 5 的平方...
计算结果: [1, 4, 9, 16, 25]
```

**要点：**  
✅ **使用 `multiprocessing.Pool` 实现多进程任务**  
✅ **提升计算密集型任务的执行效率**  

---

**总结：**  
本周练习了 **SQLite 数据库操作、Scrapy 爬虫、多进程优化**，如果有问题欢迎提问！🚀

### **Python 第三个月第三周练习（量化交易 & AI 预测 & 网络爬虫进阶）**  

**目标：**  
掌握 **Freqtrade 进行量化交易策略开发、LSTM 进行时间序列预测、Scrapy 分布式爬取数据**，提升 Python 在 **金融量化、机器学习预测、网络数据采集** 方面的能力。  

---

## **练习 1：使用 Freqtrade 进行量化交易策略开发**  
**任务：**  
- **编写一个 Freqtrade 策略**，使用 **均线交叉（SMA）** 进行买卖信号判断。  

### **步骤 1：安装 Freqtrade**
```bash
pip install freqtrade
freqtrade new-config --config config.json
```

### **步骤 2：编写策略**
创建策略文件 `user_data/strategies/SmaStrategy.py`：

```python
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import IntParameter
import pandas as pd

class SmaStrategy(IStrategy):
    minimal_roi = {"0": 0.10}
    stoploss = -0.05
    timeframe = "5m"

    sma_short = IntParameter(5, 20, default=10)
    sma_long = IntParameter(20, 50, default=30)

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["sma_short"] = dataframe["close"].rolling(self.sma_short.value).mean()
        dataframe["sma_long"] = dataframe["close"].rolling(self.sma_long.value).mean()
        return dataframe

    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (dataframe["sma_short"] > dataframe["sma_long"]),
            "buy"] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (dataframe["sma_short"] < dataframe["sma_long"]),
            "sell"] = 1
        return dataframe
```

### **步骤 3：运行回测**
```bash
freqtrade backtest --strategy SmaStrategy
```

**要点：**  
✅ **使用 `rolling().mean()` 计算均线**  
✅ **当短周期均线 `sma_short` 上穿 `sma_long` 时买入**  
✅ **当短周期均线下穿 `sma_long` 时卖出**  

---

## **练习 2：使用 LSTM 预测股票价格（时间序列预测）**  
**任务：**  
- **使用 PyTorch 训练一个 LSTM 模型**，预测股票价格。  

### **步骤 1：安装 PyTorch**
```bash
pip install torch torchvision pandas numpy matplotlib
```

### **步骤 2：训练 LSTM 预测模型**
```python
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 读取数据
df = pd.read_csv("stock_prices.csv")
data = df["Close"].values.reshape(-1, 1)

# 归一化
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# 创建时间序列数据
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 10
X, y = create_sequences(data, seq_length)

# 转换为 Tensor
X_train = torch.tensor(X[:-10], dtype=torch.float32)
y_train = torch.tensor(y[:-10], dtype=torch.float32)

# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

model = LSTMModel(input_size=1, hidden_size=50, output_size=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# 训练模型
for epoch in range(100):
    optimizer.zero_grad()
    output = model(X_train)
    loss = loss_fn(output, y_train)
    loss.backward()
    optimizer.step()

print("训练完成")

# 预测
X_test = torch.tensor(X[-10:], dtype=torch.float32)
predicted = model(X_test).detach().numpy()
predicted = scaler.inverse_transform(predicted)

plt.plot(df["Close"].values[-20:], label="真实价格")
plt.plot(range(10, 20), predicted, label="预测价格", linestyle="dashed")
plt.legend()
plt.show()
```

**要点：**  
✅ **使用 LSTM 进行时间序列预测**  
✅ **`MinMaxScaler()` 归一化数据**  
✅ **使用 `nn.LSTM()` 训练预测模型**  

---

## **练习 3：使用 Scrapy-Redis 进行分布式爬虫**  
**任务：**  
- **使用 Scrapy-Redis 进行分布式爬取商品数据**，存入 Redis。  

### **步骤 1：安装 Scrapy-Redis**
```bash
pip install scrapy scrapy-redis
```

### **步骤 2：创建 Scrapy 爬虫**
创建 `spiders/products.py`：

```python
import scrapy
from scrapy_redis.spiders import RedisSpider

class ProductsSpider(RedisSpider):
    name = "products"
    redis_key = "products:start_urls"

    def parse(self, response):
        for product in response.css(".product"):
            yield {
                "name": product.css(".title::text").get(),
                "price": product.css(".price::text").get(),
            }
```

### **步骤 3：配置 Redis**
在 `settings.py` 添加：
```python
REDIS_URL = "redis://localhost:6379"
DUPEFILTER_CLASS = "scrapy_redis.dupefilter.RFPDupeFilter"
SCHEDULER = "scrapy_redis.scheduler.Scheduler"
SCHEDULER_PERSIST = True
```

### **步骤 4：启动 Redis 并运行爬虫**
```bash
redis-server
scrapy crawl products
```

**要点：**  
✅ **使用 `scrapy_redis.spiders.RedisSpider` 进行分布式爬取**  
✅ **数据存入 Redis 进行去重**  
✅ **适用于大规模数据采集**  

---

### **总结**
✅ **Freqtrade 量化策略开发（均线交叉）**  
✅ **PyTorch LSTM 进行股票价格预测**  
✅ **Scrapy-Redis 分布式爬取商品数据**  

🔥 **本周练习涉及金融量化、机器学习、网络爬虫三大方向，适合进阶学习！**

### **Python 练习：每天自动爬取天气信息，并邮件提醒用户**  

**目标：**  
- **使用 `requests` 爬取天气信息**  
- **使用 `smtplib` 发送邮件**  
- **使用 `schedule` 设置定时任务**  

---

## **步骤 1：安装必要的库**
```bash
pip install requests schedule
```

---

## **步骤 2：获取天气信息**
我们使用 **API** 获取天气信息，例如 **OpenWeatherMap** 或 **某些天气网站爬取**。

```python
import requests

def get_weather(city="Beijing"):
    api_key = "your_api_key"  # 需要替换成你的API密钥
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        weather = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        return f"今日 {city} 天气：{weather}, 气温：{temp}°C"
    else:
        return "获取天气失败，请检查 API Key。"

# 测试获取天气
print(get_weather("Shanghai"))
```

**替代方案（使用爬虫获取天气）**  
如果没有 API，可以用 **requests + BeautifulSoup 爬取天气网站**：
```python
from bs4 import BeautifulSoup

def get_weather_alternative():
    url = "https://www.weather.com/weather/today/l/CHXX0008:1:CH"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    soup = BeautifulSoup(response.text, "html.parser")
    weather = soup.find(class_="CurrentConditions--phraseValue--2xXSr").text
    temp = soup.find(class_="CurrentConditions--tempValue--3KcTQ").text
    return f"今日天气：{weather}, 气温：{temp}"

# 测试爬取天气
print(get_weather_alternative())
```

---

## **步骤 3：发送邮件**
使用 `smtplib` 发送邮件，支持 **Gmail / QQ 邮箱 / 163 邮箱**。

```python
import smtplib
from email.mime.text import MIMEText
from email.header import Header

# 配置邮件发送
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "your_email@gmail.com"
SENDER_PASSWORD = "your_password"
RECEIVER_EMAIL = "receiver_email@example.com"

def send_email(subject, message):
    msg = MIMEText(message, "plain", "utf-8")
    msg["From"] = Header("天气提醒", "utf-8")
    msg["To"] = Header("用户", "utf-8")
    msg["Subject"] = Header(subject, "utf-8")

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()
        print("邮件发送成功！")
    except Exception as e:
        print("邮件发送失败：", e)

# 测试邮件
send_email("天气提醒", get_weather("Shanghai"))
```
**📌 注意：**
- 如果使用 **QQ 邮箱**，SMTP 服务器是 `"smtp.qq.com"`，需要在 **QQ 邮箱设置** 开启 **SMTP** 并获取授权码。  
- **163 邮箱** 服务器是 `"smtp.163.com"`，同样需要授权码。  

---

## **步骤 4：定时任务**
使用 `schedule` 让脚本 **每天早上 8 点自动运行**：

```python
import schedule
import time

def job():
    city = "Shanghai"
    weather_info = get_weather(city)
    send_email(f"{city} 天气提醒", weather_info)

# 设置每天早上 8 点运行任务
schedule.every().day.at("08:00").do(job)

while True:
    schedule.run_pending()
    time.sleep(60)  # 每分钟检查一次
```

---

## **最终效果**
✅ **每天早上 8 点自动爬取天气信息**  
✅ **自动发送天气提醒邮件**  
✅ **支持 API 或爬取天气网站**  

---

💡 **可以扩展：**  
- **支持多个城市天气提醒**  
- **结合 `telegram` 机器人推送天气**  
- **存储天气历史数据，做趋势分析**  

🚀 **现在你可以试试运行代码，定期收到天气邮件提醒！**