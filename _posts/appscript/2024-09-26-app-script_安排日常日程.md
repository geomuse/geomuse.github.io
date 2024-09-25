---
layout: post
title:  app script 安排日常日程
date:   2024-09-26 11:24:29 +0800
categories: 
    - script 
---

```js
// Google Calendar 的 ID (可以是 'primary' 代表默认日历)
var calendarId = 'primary';  // 或替换为其他日历 ID
var calendar = CalendarApp.getCalendarById(calendarId);

// 读取 Google Sheets 中的任务并创建日历事件
function createTaskEvents() {
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
  
  // 获取所有数据
  var data = sheet.getDataRange().getValues();

  // 遍历每一行数据，从第二行（跳过标题）
  for (var i = 1; i < data.length; i++) {
    var taskTitle = data[i][0];  // 任务名称
    var taskDate = data[i][1];   // 任务日期
    var startTime = data[i][2];  // 开始时间
    var endTime = data[i][3];    // 结束时间
    var taskDescription = data[i][4]; // 描述（可选）
    var priority = data[i][5];   // 优先级（可选）
    
    // 合并日期和时间生成事件的开始和结束时间
    var startDateTime = new Date(taskDate + " " + startTime);
    var endDateTime = new Date(taskDate + " " + endTime);

    // 创建 Google Calendar 事件
    var event = calendar.createEvent(taskTitle, startDateTime, endDateTime)
                        .setDescription(taskDescription)
                        .addEmailReminder(10);  // 可根据需要设置提醒（如提前10分钟）

    // 打印日志，方便查看任务创建情况
    Logger.log('创建了任务事件：' + taskTitle + '，优先级：' + priority);
  }
}
```