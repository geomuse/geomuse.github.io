---
layout: post
title:  app script 建立清单定时发送
date:   2024-09-28 11:24:29 +0800
categories: 
    - script 
---

针对定时处理

```js
function sendDocAsPDFAsBooks() {
  // 替换为你的 Google Docs 文档 ID
  var docId = '1TftqVxVVqzdB31hI6zWcjvsSVJNwDutb2NaKTWcRMhk';
  
  // 替换为接收邮件的地址
  var emailAddress = 'boonhong565059@gmail.com';
  
  // 获取文档
  var doc = DriveApp.getFileById(docId);
  
  // 将文档转换为 PDF
  var pdfBlob = doc.getAs('application/pdf');
  
  // 设置邮件主题和正文
  var subject = '读书清单';
  var body = '这件邮件是为了解决每日读书清单被提醒而建立的.';
  
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

针对定时处理

```js
function sendDocAsPDFAsSport() {
  // 替换为你的 Google Docs 文档 ID
  var docId = '1iZPMchuERe4EBX55IRA9luUfDgr2Z_rUNQ1inRFYFvU';
  
  // 替换为接收邮件的地址
  var emailAddress = 'boonhong565059@gmail.com';
  
  // 获取文档
  var doc = DriveApp.getFileById(docId);
  
  // 将文档转换为 PDF
  var pdfBlob = doc.getAs('application/pdf');
  
  // 设置邮件主题和正文
  var subject = '健身清单';
  var body = '这件邮件是为了解决每日健身清单被提醒而建立的.';
  
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