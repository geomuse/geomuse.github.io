---
layout: post
title: flutter 从A页跳到B页并回传资料
date: 2024-09-23 11:24:29 +0800
categories:
    - flutter
---

一個是 `MyHomePage`（首頁），

另一個是 `BPage` (Ｂ頁)。

透過 `Navigator.push` 方法從首頁跳到`Ｂ`頁，

並傳定兩種資料型態過去，

一個是整數資料，另一種是字串資料。

```dart
import 'package:flutter/material.dart';
import 'b_page.dart';

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
      ),
    );
  }
}

class HomePage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Center(
      child: ElevatedButton(
        child: Text('跳到 B 頁'),
        onPressed: () {
          Navigator.push(
              context,
              MaterialPageRoute(
                  builder: (context) => BPage(intVal: 100, strVal: 'HKT線上教室')));
        },
      ),
    );
  }
}
```

`b_dart.dart`

```dart
import 'package:flutter/material.dart';
class BPage extends StatelessWidget {
  int intVal=0;
  String strVal="";

  BPage({Key? key, this.intVal=0, this.strVal=""}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('我是 B 頁'),
      ),
      body: Center(
        child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: <Widget>[
              Text('intVal: $intVal'),
              Text('strVal: $strVal')
            ]),
      ),
    );
  }
}
```