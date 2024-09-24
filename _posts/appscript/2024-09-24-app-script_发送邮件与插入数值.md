---
layout: post
title:  app script 发送邮件与插入数值
date:   2024-09-24 11:24:29 +0800
categories: 
    - app script 
---

把文档发送邮件到邮件上

```js
function myFunction() {
  let excel = SpreadsheetApp.getActiveSpreadsheet();
  let sheet = excel.getActiveSheet();
  let meeting_notifaction = DocumentApp.openById('1-k_-cQFasIDYJRxalCNMBg24A1epDMvJHnoJg5yuCQw') ; 
  let content = meeting_notifaction.getBody().getText() ; 
  GmailApp.sendEmail('boonhong565059@gmail.com','title',content) ;
}
```

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