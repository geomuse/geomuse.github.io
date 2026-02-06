---
layout: post
title:  nodejs express respone
date:   2026-02-14 09:01:00 +0800
tags: 
    - nodejs
image: 16.jpg
---

```js
app.get('/404',(req,res)=>{
    res.redirect('/')
});
```

完整代码

```js
const express = require('express');
const app = express();

app.get('/', (req, res) => {
    res.send('Hello World');
});

app.get('/user', (req, res) => {
    res.send('User Page');
});

app.listen(3000, () => {
    console.log('Server is running on port 3000');
});
```