---
layout: post
title:  app script 安排日常日程
date:   2024-09-26 11:24:29 +0800
categories: 
    - script 
---

```js
function sendTasksToGoogleCalendar() {
  // 获取当前活跃的 Sheet
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
  var dataRange = sheet.getDataRange();
  var data = dataRange.getValues();
  // 获取 Google Calendar 的对象 (指定的日历ID，如果是默认日历可以用 getDefaultCalendar)
  var calendar = CalendarApp.getDefaultCalendar();
  
  // 从第二行开始读取，假设第一行是标题
  for (var i = 1; i < data.length; i++) {
    var taskName = data[i][0];  // 任务名称
    var taskDescription = data[i][1];  // 任务描述
    var taskDate = new Date(data[i][2]);  // 任务日期
    var taskTime = data[i][3];  // 任务时间
    // 将时间字符串转化为日期对象
    var startTime = new Date(taskDate);
    startTime.setHours(taskTime.getHours());
    startTime.setMinutes(taskTime.getMinutes());
    // 创建日历事件
    calendar.createEvent(taskName, startTime, startTime, {
      description: taskDescription
    });
    // 可选：在 Sheets 中标记任务已发送
    sheet.getRange(i + 1, 6).setValue('已发送');
  }
}
```