---
layout: post
title: flutter route navigator
date: 2024-09-22 11:24:29 +0800
categories:
    - flutter
---

**路由与导航** 系统。通过路由和导航，应用可以在多个页面之间进行切换，这是构建复杂 Flutter 应用的基础。

### 1. **什么是路由和导航？**
在 Flutter 中，页面通常被称为“路由（Route）”，而“导航（Navigator）”是负责在这些路由之间进行页面切换的系统。我们可以通过导航器来执行页面的推入、移除、替换等操作。

#### Navigator 类常用的方法：
- **push**: 将一个新页面推入路由堆栈，并显示该页面。
- **pop**: 移除当前页面（顶层页面），返回到上一个页面。
- **pushReplacement**: 将当前页面替换为新页面，不能返回到之前的页面。

### 2. **基本的页面跳转**

我们首先来实现一个简单的页面跳转示例，使用 `Navigator.push` 和 `Navigator.pop` 方法。

#### 示例代码：基本页面跳转
```dart
import 'package:flutter/material.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: FirstPage(),
    );
  }
}

class FirstPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('第一页'),
      ),
      body: Center(
        child: ElevatedButton(
          child: Text('跳转到第二页'),
          onPressed: () {
            Navigator.push(
              context,
              MaterialPageRoute(builder: (context) => SecondPage()),
            );
          },
        ),
      ),
    );
  }
}

class SecondPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('第二页'),
      ),
      body: Center(
        child: ElevatedButton(
          child: Text('返回第一页'),
          onPressed: () {
            Navigator.pop(context); // 返回到上一页
          },
        ),
      ),
    );
  }
}
```

#### 解释：
- **Navigator.push**: 将第二页推入到路由栈，显示该页面。
- **Navigator.pop**: 将第二页从路由栈中移除，返回到第一页。
- **MaterialPageRoute**: Flutter 提供的一个路由类，用于创建页面之间的过渡动画。

### 3. **命名路由**

在大型应用中，页面跳转可能涉及多个页面。使用**命名路由**可以更方便地管理这些页面。我们可以为每个页面指定一个路由名称，在需要时通过路由名称来进行跳转。

#### 示例代码：命名路由
```dart
import 'package:flutter/material.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      // 定义命名路由
      routes: {
        '/': (context) => FirstPage(), // 默认页面
        '/second': (context) => SecondPage(),
      },
      initialRoute: '/', // 启动时显示的页面
    );
  }
}

class FirstPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('第一页'),
      ),
      body: Center(
        child: ElevatedButton(
          child: Text('跳转到第二页'),
          onPressed: () {
            Navigator.pushNamed(context, '/second');
          },
        ),
      ),
    );
  }
}

class SecondPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('第二页'),
      ),
      body: Center(
        child: ElevatedButton(
          child: Text('返回第一页'),
          onPressed: () {
            Navigator.pop(context); // 返回到上一页
          },
        ),
      ),
    );
  }
}
```

#### 解释：
- **routes**: 定义了应用中的命名路由，每个路由名称对应一个页面组件。
- **Navigator.pushNamed**: 使用路由名称进行页面跳转。

### 4. **传递参数**

在实际应用中，我们通常需要在页面之间传递数据。Flutter 支持在页面跳转时传递参数。

#### 示例代码：传递参数
```dart
import 'package:flutter/material.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      routes: {
        '/': (context) => FirstPage(),
        '/second': (context) => SecondPage(),
      },
    );
  }
}

class FirstPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('第一页'),
      ),
      body: Center(
        child: ElevatedButton(
          child: Text('跳转到第二页并传递参数'),
          onPressed: () {
            Navigator.push(
              context,
              MaterialPageRoute(
                builder: (context) => SecondPage(data: '这是从第一页传递的参数'),
              ),
            );
          },
        ),
      ),
    );
  }
}

class SecondPage extends StatelessWidget {
  final String data;

  SecondPage({required this.data});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('第二页'),
      ),
      body: Center(
        child: Text(data), // 显示传递过来的参数
      ),
    );
  }
}
```

#### 解释：
- **传递参数**: 在创建 `SecondPage` 时，通过构造函数将参数传递过去。页面可以接收并显示这些参数。
- **Navigator.push**: 通过 `MaterialPageRoute` 传递参数给目标页面。

### 5. **从页面返回数据**

有时我们需要在从一个页面返回时传递数据回去，类似于在一个页面中选择数据并返回到上一个页面。

#### 示例代码：从页面返回数据
```dart
import 'package:flutter/material.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: FirstPage(),
    );
  }
}

class FirstPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('第一页'),
      ),
      body: Center(
        child: ElevatedButton(
          child: Text('跳转到第二页并选择数据'),
          onPressed: () async {
            // 等待从第二页返回的结果
            final result = await Navigator.push(
              context,
              MaterialPageRoute(builder: (context) => SecondPage()),
            );

            // 显示返回结果
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(content: Text('返回的数据是: $result')),
            );
          },
        ),
      ),
    );
  }
}

class SecondPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('第二页'),
      ),
      body: Center(
        child: ElevatedButton(
          child: Text('选择数据并返回'),
          onPressed: () {
            // 返回数据到上一页
            Navigator.pop(context, '选择的数据');
          },
        ),
      ),
    );
  }
}
```

#### 解释：
- **Navigator.push**: 页面跳转并等待返回值。
- **Navigator.pop**: 返回时通过第二个参数将数据传回前一个页面。
- **ScaffoldMessenger**: 用于显示 `SnackBar` 来提示用户选择的结果。

### 6. **pushReplacement 和 pushAndRemoveUntil**

Flutter 提供了其他导航方法，来替换当前页面或清除多个页面。常见的有：
- **Navigator.pushReplacement**: 替换当前页面，不能返回。
- **Navigator.pushAndRemoveUntil**: 将新页面推入，并移除堆栈中所有指定的页面。

#### 示例代码：pushReplacement
```dart
Navigator.pushReplacement(
  context,
  MaterialPageRoute(builder: (context) => NewPage()),
);
```

#### 示例代码：pushAndRemoveUntil
```dart
Navigator.pushAndRemoveUntil(
  context,
  MaterialPageRoute(builder: (context) => NewPage()),
  (route) => false,  // 移除所有页面
);
```