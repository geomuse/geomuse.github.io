---
layout: post
title:  nodejs express template engine
date:   2026-02-16 09:01:00 +0800
tags: 
    - nodejs
image: 16.jpg
---

```bash
npm install ejs
```

```js
const express = require('express');
const app = express();

app.set("view engine",'ejs');
app.set("views",'./views');

app.get('/', (req, res) => {
    res.send('Hello World');
});

app.get('/user', (req, res) => {
    const city = req.query.city;
    if(city == 'taipei'){
        let data = { 'pop': 30000000, 'name': 'taipei' };
        // 关键点：将 data 作为第二个参数传入
        res.render("city", { data: data }); 
    } 
    else {
        res.send('only taipei');
    }
});

app.get('/404',(req,res)=>{
    res.redirect('/')
});

app.listen(3000, () => {
    console.log('Server is running on port 3000');
});
```