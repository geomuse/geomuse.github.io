---
layout: post
title:  app script 发送邮件
date:   2024-09-25 11:24:29 +0800
categories: 
    - script 
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

发送邮件

```js
let content = [
  {"email": "boonhong565059@gmail.com", "message": "Hello User 1"},
  {"email": "boonhong565059@gmail.com", "message": "Hello User 2"}
]

function sendEmailsFromExternalJson(){
  for (let i = 0; i < content.length; i++) {
    let recipient = content[i].email ; 
    let message = content[i].message;
    console.log(message);  // 输出 message
    GmailApp.sendEmail(recipient, 'Automated Email', message);
  }
  Logger.log('Emails sent successfully.');
}
```

填写问卷后