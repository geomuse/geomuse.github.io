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
  {"email": "boonhong565059@gmail.com", "message": "本文将用通俗易懂的方式讲解分类问题的混淆矩阵和各种评估指标的计算公式。将要给大家介绍的评估指标有：准确率、精准率、召回率、F1、ROC曲线、AUC曲线"},
  {"email": "boonhong565059@gmail.com", "message": "机器学习有很多评估的指标。有了这些指标我们就横向的比较哪些模型的表现更好。我们先从整体上来看看主流的评估指标都有哪"}
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