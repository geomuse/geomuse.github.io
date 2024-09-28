---
layout: post
title:  flutter 漂浮按鈕
date:   2024-09-27 11:24:29 +0800
categories:
    - flutter
---

FloatingActionButton 漂浮按鈕，常用屬性

- onPressed：按鈕按下後觸發的方法邏輯程式。
- elevation：按鈕陰影大小值
- foregrnundColor：前景色
- backgroundColor：背景色
- floatingActionButtonLocation：設定漂浮按鈕顯示的位置
  - centerDocked 底部中間
  - endDocked 右下角
  - centerFloat 中間偏上
  - endFloat 底部偏上

```dart
import 'package:flutter/material.dart';

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: Text('HKT 線上教室'),
        ),
        body: HomePage(),
        //注意是在 Scaffold 裡面喲!!
        floatingActionButtonLocation: FloatingActionButtonLocation.endFloat,
        floatingActionButton: FloatingActionButton(
          child: Icon(Icons.add),
          onPressed: () {
            print('press...');
          },
        ),
      ),
    );
  }
}

class HomePage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Center(
      child: Text('漂浮按鈕'),
    );
  }
}
```

```dart
foregroundColor: Colors.amber,
backgroundColor: Colors.red,
```

```dart
floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat
```