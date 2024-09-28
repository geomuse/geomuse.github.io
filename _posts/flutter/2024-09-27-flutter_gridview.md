---
layout: post
title:  flutter GridView
date:   2024-09-27 11:24:29 +0800
categories:
    - flutter
---

自定義顯示三列的 GridView 網格佈局

```dart
import 'package:flutter/material.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
        home: Scaffold(
      appBar: AppBar(
        title: Text('HKT線上教室'),
      ),
      body: HomePage(),
    ));
  }
}

class HomePage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return new GridView.count(
      crossAxisCount: 3,
      children: new List.generate(50, (index) {
        return new Center(
          child: new Text(
            '測試資料 $index',
          ),
        );
      }),
    );
  }
}
```