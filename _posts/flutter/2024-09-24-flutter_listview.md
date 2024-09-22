---
layout: post
title: flutter ListView
date: 2024-09-24 11:24:29 +0800
categories:
    - flutter
---

ListView 和 ListView.builder 最大的差異是 ListView 是一次建立完所有列表資料，如果記憶體沒有管理好很容易造成APP閃退，

ListView.builder 則是當元件滾動到螢幕當下才進行創建。

错误范例 : 使用 ListView 垂直顯示「自定義樣式」列表資料

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
    return Center(
      //列表元件
      child: ListView(
        children: <Widget>[
          Row(
            children: <Widget>[Text('測試資料'), Text('測試資料'), Text('測試資料')],
          ),
          Column(
            children: <Widget>[Text('測試資料'), Text('測試資料'), Text('測試資料')],
          ),
          Row(
            children: <Widget>[Text('測試資料'), Text('測試資料'), Text('測試資料')],
          )
        ],
      ),
    );
  }
}
```

使用 ListView.builder 垂直顯示「大量」列表資料

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
  //透過資料產生器，產生資料
  final List<Product> listItems = new List<Product>.generate(500, (i) {
    return Product(
      name: '測試資料 $i',
      price: '售價：$i',
    );
  });

  @override
  Widget build(BuildContext context) {
    return Center(
        child: ListView.builder(
      itemCount: listItems.length,
      itemBuilder: (context, index) {
        return ListTile(
          leading: Icon(Icons.event_seat),
          title: Text('${listItems[index].name}'),
          subtitle: Text('${listItems[index].price}'),
        );
      },
    ));
  }
}

//產品資料
class Product {
  final String name;
  final String price;

  Product({this.name, this.price});
}
```