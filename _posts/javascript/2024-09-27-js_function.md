---
layout: post
title:  java script function
date:   2024-09-27 11:24:29 +0800
categories: 
    - script
---

在前端

```pug
ul#books
```

在 `JS`

```js
for(var i=0 ; i < bookdatas.length ; i++){
    var book = bookdatas[i]
    $('ul#books').append("<li>"+book.name+"</li>")
}
```