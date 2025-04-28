---
layout: post
title:  app script 发送试算表到邮件
date:   2024-09-27 11:24:29 +0800
categories: 
    - script 
---

发送试算表到邮件

```js
function sendEmailassheets() {
  // spread list.
  var spreadsheetId = '1J_oupx9tWFdMQh3Au3S_cByQxTOmOl9zalVTLnBuQVA';
  var file = DriveApp.getFileById(spreadsheetId);
  var emailAddress = 'boonhong565059@gmail.com';
  var subject = '日常行程规划试算表';
  var body = '';
  
  // 设置导出选项
  var url = 'https://docs.google.com/spreadsheets/d/' + spreadsheetId + '/export?';
  var exportOptions = {
    // 在这里添加导出参数
  };
  
  // 生成完整的导出 URL
  var params = [];
  for (var param in exportOptions) {
    params.push(param + '=' + exportOptions[param]);
  }
  var exportUrl = url + params.join('&');
  
  // 获取 PDF 内容
  var response = UrlFetchApp.fetch(exportUrl, {
    headers: {
      'Authorization': 'Bearer ' + ScriptApp.getOAuthToken(),
    },
  });
  var pdfContent = response.getBlob();
  
  MailApp.sendEmail({
    to: emailAddress,
    subject: subject,
    body: body,
    attachments: [pdfContent]
  });
  
  Logger.log('已经发送至 ' + emailAddress);
}
```