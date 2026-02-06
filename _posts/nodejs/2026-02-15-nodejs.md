---
layout: post
title:  nodejs express static files
date:   2026-02-15 09:01:00 +0800
tags: 
    - nodejs
image: 16.jpg
---

```js
const express = require('express');
const app = express();

app.use(express.static(path.join(__dirname, 'index')));

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