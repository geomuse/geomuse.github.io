---
layout: post
title:  nodejs express http requests
date:   2026-02-12 09:01:00 +0800
tags: 
    - nodejs
image: 16.jpg
---

```js
app.get('/', (req, res) => {
    res.send('Hello World');
});
```

req 使用方法

```js
console.log(req.get("accept-language"));
```

完整代码

```js
const express = require('express');
const app = express();

app.get('/', (req, res) => {
    console.log(req.get("accept-language"));
    res.send('Hello World');
});


app.listen(3000, () => {
    console.log('Server is running on port 3000');
});
```
