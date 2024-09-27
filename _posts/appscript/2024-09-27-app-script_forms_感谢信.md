---
layout: post
title:  app script 建立问卷后感谢信
date:   2024-09-29 11:24:29 +0800
categories: 
    - script 
---

```js
function onFormSubmitAfterEmail(e) {
  // 获取表单响应
  var response = e.response;
  
  // 获取所有回答
  var itemResponses = response.getItemResponses();
  
  // 初始化电子邮件变量
  var emailAddress = "boonhong565059@gmail.com";
  
  // 遍历所有回答，找到电子邮件地址
  for (var i = 0; i < itemResponses.length; i++) {
    var itemResponse = itemResponses[i];
    var question = itemResponse.getItem().getTitle();
    var answer = itemResponse.getResponse();
    
    // 替换“电子邮件”为您表单中收集电子邮件的实际问题标题
    if (question === "电子邮件") {
      emailAddress = answer;
      break;
    }
  }
  // 如果找到电子邮件地址，发送感谢信
  if (emailAddress) {
    var subject = "感谢您参与我们的问卷调查";
    var body = "亲爱的参与者，\n\n感谢您抽出宝贵的时间填写我们的问卷。您的反馈对我们非常重要！\n\n祝好，\n您的团队";
    
    MailApp.sendEmail(emailAddress, subject, body);
  }
}
```