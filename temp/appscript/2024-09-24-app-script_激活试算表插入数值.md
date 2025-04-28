---
layout: post
title:  app script 激活试算表插入数值
date:   2024-09-24 11:24:29 +0800
categories: 
    - script 
---

插入数值到`sheet`里的`A1`,`A2`

```js
function myFunction() {
  let a = 1 
  let b = 2
  let excel = SpreadsheetApp.getActiveSpreadsheet() ; 
  let sheet = excel.getActiveSheet() ;
  
  sheet.getRange(1,1).setValue(a) ;
  sheet.getRange(2,1).setValue(b) ;
}
```