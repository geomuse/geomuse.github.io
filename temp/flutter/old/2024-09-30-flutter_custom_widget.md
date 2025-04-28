---
layout: post
title:  flutter custom widget
date:   2024-09-30 11:24:29 +0800
categories:
    - flutter
---

**自定义 Widget**，这是构建复杂和可重用用户界面的关键步骤。通过自定义 Widget，你可以实现独特的 UI 组件，提升应用的可维护性和扩展性。

### 1. **什么是自定义 Widget？**
在 Flutter 中，几乎所有内容都是 Widget，包括按钮、文本、布局等。通过创建自定义 Widget，可以将复杂的界面逻辑封装起来，从而更容易复用和管理。Flutter 支持两种主要类型的自定义 Widget：

- **StatelessWidget**：无状态，适用于不需要动态更新数据的组件。
- **StatefulWidget**：有状态，适用于需要根据用户交互或外部数据进行动态更新的组件。

### 2. **创建 StatelessWidget**

`StatelessWidget` 是不需要维护内部状态的组件，常用于显示静态内容。我们可以通过继承 `StatelessWidget` 类来创建自定义的静态组件。

#### 示例代码：自定义 StatelessWidget
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
        appBar: AppBar(title: Text('自定义 StatelessWidget')),
        body: Center(
          child: CustomButton(
            text: '点击我',
            onPressed: () {
              print('按钮被点击');
            },
          ),
        ),
      ),
    );
  }
}

// 自定义无状态按钮组件
class CustomButton extends StatelessWidget {
  final String text;
  final VoidCallback onPressed;

  CustomButton({required this.text, required this.onPressed});

  @override
  Widget build(BuildContext context) {
    return ElevatedButton(
      onPressed: onPressed,
      child: Text(text),
      style: ElevatedButton.styleFrom(
        padding: EdgeInsets.symmetric(horizontal: 20, vertical: 10),
        textStyle: TextStyle(fontSize: 20),
      ),
    );
  }
}
```

#### 解释：
- **CustomButton**：自定义的无状态按钮组件，通过 `text` 参数设置按钮文字，通过 `onPressed` 回调处理点击事件。
- **ElevatedButton**：基础的按钮组件，封装在自定义的 `CustomButton` 中，使其更灵活可复用。

### 3. **创建 StatefulWidget**

`StatefulWidget` 用于创建需要动态更新的组件。例如，计数器、表单输入等组件都需要根据用户交互实时更新 UI。通过继承 `StatefulWidget` 类并实现 `State` 类，你可以实现有状态的自定义组件。

#### 示例代码：自定义 StatefulWidget
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
        appBar: AppBar(title: Text('自定义 StatefulWidget')),
        body: Center(
          child: CustomCounter(),
        ),
      ),
    );
  }
}

// 自定义有状态计数器组件
class CustomCounter extends StatefulWidget {
  @override
  _CustomCounterState createState() => _CustomCounterState();
}

class _CustomCounterState extends State<CustomCounter> {
  int _counter = 0;

  void _incrementCounter() {
    setState(() {
      _counter++;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Text(
          '计数值: $_counter',
          style: TextStyle(fontSize: 24),
        ),
        SizedBox(height: 20),
        ElevatedButton(
          onPressed: _incrementCounter,
          child: Text('增加计数'),
        ),
      ],
    );
  }
}
```

#### 解释：
- **CustomCounter**：自定义的有状态计数器组件，内部维护 `_counter` 状态，每次点击按钮时通过 `setState` 更新计数值。
- **StatefulWidget**：通过 `State` 类来管理组件的状态，并通过 `setState` 方法来触发 UI 更新。

### 4. **组件复用与参数传递**

自定义 Widget 的核心思想之一是提高复用性和灵活性。通过将参数传递给自定义组件，可以使得同一个组件在不同场景下表现出不同的行为。

#### 示例代码：带参数的自定义组件
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
        appBar: AppBar(title: Text('参数化自定义组件')),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              CustomCard(
                color: Colors.blue,
                text: '蓝色卡片',
              ),
              CustomCard(
                color: Colors.green,
                text: '绿色卡片',
              ),
              CustomCard(
                color: Colors.red,
                text: '红色卡片',
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// 自定义卡片组件
class CustomCard extends StatelessWidget {
  final Color color;
  final String text;

  CustomCard({required this.color, required this.text});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: EdgeInsets.all(10),
      padding: EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: color,
        borderRadius: BorderRadius.circular(10),
      ),
      child: Text(
        text,
        style: TextStyle(color: Colors.white, fontSize: 18),
      ),
    );
  }
}
```

#### 解释：
- **CustomCard**：自定义的卡片组件，通过 `color` 和 `text` 参数控制不同卡片的背景颜色和显示文本。
- **复用组件**：在 `MyApp` 中我们创建了多个 `CustomCard`，每个卡片的外观和内容不同，但都复用了相同的逻辑。

### 5. **组合与分解 Widget**

Flutter 鼓励通过组合现有的 Widget 来构建复杂的 UI。通过将小的、自定义的组件组合在一起，你可以构建出复杂的界面。另一方面，也可以通过将复杂组件拆分为多个小组件来提高代码的可读性和维护性。

#### 示例代码：组合 Widget
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
        appBar: AppBar(title: Text('组合 Widget')),
        body: Center(
          child: ProfileCard(
            name: 'John Doe',
            age: 28,
            imageUrl: 'https://example.com/profile.jpg',
          ),
        ),
      ),
    );
  }
}

// 自定义的组合组件
class ProfileCard extends StatelessWidget {
  final String name;
  final int age;
  final String imageUrl;

  ProfileCard({required this.name, required this.age, required this.imageUrl});

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            CircleAvatar(
              radius: 40,
              backgroundImage: NetworkImage(imageUrl),
            ),
            SizedBox(height: 10),
            Text(
              name,
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
            Text('年龄: $age'),
          ],
        ),
      ),
    );
  }
}
```

#### 解释：
- **ProfileCard**：这是一个组合的自定义组件，将头像、姓名和年龄等信息组合成一个用户卡片。
- **组合**：我们通过 `CircleAvatar`、`Text`、`Card` 等现有的 Flutter 组件来创建更复杂的 `ProfileCard` 组件。