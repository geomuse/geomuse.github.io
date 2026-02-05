---
layout: post
title:  nodejs express 快速入门
date:   2026-02-10 09:01:00 +0800
tags: 
    - nodejs
image: 16.jpg
---

```bash
npm install express
```

```js
const express = require('express');
const app = express();

app.get('/', (req, res) => {
    res.send('Hello World');
});

app.listen(3000, () => {
    console.log('Server is running on port 3000');
});
```