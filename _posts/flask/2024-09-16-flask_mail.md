---
layout: post
title:  challenge-25/30-Flask-Mail
date:   2024-09-16 11:24:29 +0800
categories: 
    - python 
    - flask
    - challenge
---

要使用 Flask-Mail 发送邮件，你可以按照以下步骤来安装并配置 Flask-Mail 扩展，并学习如何发送邮件：

### 1. 安装 Flask-Mail
首先，在你的虚拟环境中安装 `Flask-Mail` 扩展：

```bash
pip install Flask-Mail
```

### 2. 配置 Flask-Mail
在 Flask 应用中，配置邮件服务器的相关参数。常见的邮件服务提供商有 Gmail、Yahoo、Outlook 等。以 Gmail 为例，你需要在 `config.py` 中添加以下配置：

```python
# config.py

import os

class Config:
    MAIL_SERVER = 'smtp.gmail.com'  # 邮件服务器地址
    MAIL_PORT = 587  # 端口号
    MAIL_USE_TLS = True  # 启用TLS
    MAIL_USERNAME = os.getenv('MAIL_USERNAME')  # 发件人邮箱
    MAIL_PASSWORD = os.getenv('MAIL_PASSWORD')  # 发件人邮箱的密码或App密码
    MAIL_DEFAULT_SENDER = os.getenv('MAIL_DEFAULT_SENDER')  # 默认发件人
```

确保你在 `.env` 文件或环境变量中设置 `MAIL_USERNAME`, `MAIL_PASSWORD`, 和 `MAIL_DEFAULT_SENDER`：

```bash
# .env
MAIL_USERNAME='your-email@gmail.com'
MAIL_PASSWORD='your-email-password'  # 使用应用专用密码
MAIL_DEFAULT_SENDER='your-email@gmail.com'
```

### 3. 初始化 Flask-Mail

在你的 Flask 应用中初始化 Flask-Mail 扩展：

```python
from flask import Flask
from flask_mail import Mail

app = Flask(__name__)
app.config.from_object('config.Config')  # 从配置文件加载配置

mail = Mail(app)  # 初始化 Flask-Mail
```

### 4. 发送邮件
你可以通过 `Message` 类来构建邮件，并通过 `mail.send()` 发送邮件。下面是一个发送简单邮件的示例：

```python
from flask_mail import Message
from flask import Flask, render_template

# 发送邮件的函数
@app.route('/send_email')
def send_email():
    msg = Message('Hello from Flask-Mail',  # 主题
                  recipients=['recipient@example.com'])  # 收件人
    msg.body = 'This is a test email sent from Flask-Mail.'  # 邮件正文
    msg.html = '<h1>This is a test email sent from Flask-Mail</h1>'  # HTML格式的正文

    try:
        mail.send(msg)  # 发送邮件
        return "Email sent successfully!"
    except Exception as e:
        return f"Failed to send email: {str(e)}"
```

### 5. 测试邮件发送
运行 Flask 应用，然后访问 `/send_email` 路径，就会发送测试邮件。确认你在控制台没有错误，并检查收件箱是否成功收到邮件。

### 注意事项
1. **Gmail 安全性问题**：如果你使用的是 Gmail，请确保在 Gmail 账户中启用了 [应用专用密码](https://support.google.com/accounts/answer/185833?hl=en)，并在配置中使用此密码而不是普通登录密码。
2. **调试**：可以在开发环境中设置 `MAIL_DEBUG = True` 以便调试信息输出到控制台。
