---
layout: post
title:  flutter hello world
date:   2024-09-20 11:24:29 +0800
categories: 
    - flutter
---

显示`title`和`content`

`main.dart` 中插入以下代码 : 

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
        title: Center(child: Text('...')),
      ),
      body: HomePage(),
    ));
  }
}

class HomePage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Center(
      child: Text("I\'m geo"),
    );
  }
}
```