---
layout: post
title:  app script 文档转换pdf
date:   2024-09-27 11:24:29 +0800
categories: 
    - script 
---

被转换成pdf

```js
var docId = '18VBM7HyoUwzelhZFl2reGfGjBck-Sf2dHg1X3hvHuc8';

// 获取文档
var doc = DriveApp.getFileById(docId);
  
// 将文档转换为 PDF
var pdfBlob = doc.getAs('application/pdf');
```

完整代码 

```js
function sendDocAsPDF() {
  // 替换为你的 Google Docs 文档 ID
  var docId = '18VBM7HyoUwzelhZFl2reGfGjBck-Sf2dHg1X3hvHuc8';
  
  // 替换为接收邮件的地址
  var emailAddress = 'boonhong565059@gmail.com';
  
  // 获取文档
  var doc = DriveApp.getFileById(docId);
  
  // 将文档转换为 PDF
  var pdfBlob = doc.getAs('application/pdf');
  
  // 设置邮件主题和正文
  var subject = '';
  var body = '';
  
  // 发送邮件
  MailApp.sendEmail({
    to: emailAddress,
    subject: subject,
    body: body,
    attachments: [pdfBlob]
  });
  
  Logger.log('已经发送至 ' + emailAddress);
}
```