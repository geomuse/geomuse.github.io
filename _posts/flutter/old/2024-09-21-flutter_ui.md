---
layout: post
title: flutter UI
date: 2024-09-21 10:24:29 +0800
categories:
    - flutter
---

布局是 UI 开发的基础，通过使用布局组件（Widget），可以轻松地创建复杂的 UI 界面。

### 1. **常用布局Widget**
Flutter提供了多个布局组件，以下是几个常用的：

- **Container**: 用于容纳子组件，支持背景色、边框、阴影等属性。
- **Row**: 将子组件水平排列。
- **Column**: 将子组件垂直排列。
- **Stack**: 将子组件叠加在一起，子组件按顺序堆叠。

### 2. **Container**
`Container` 是 Flutter 中最基础的布局组件，它用于包裹子组件并可以进行样式的调整，比如设置尺寸、边距、对齐方式等。

#### 示例代码
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
          title: Text('Container 示例'),
        ),
        body: Center(
          child: Container(
            width: 200,
            height: 200,
            padding: EdgeInsets.all(20),
            margin: EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: Colors.blue,
              borderRadius: BorderRadius.circular(10),
              boxShadow: [
                BoxShadow(
                  color: Colors.black26,
                  offset: Offset(2, 2),
                  blurRadius: 10,
                )
              ],
            ),
            child: Center(
              child: Text(
                '这是一个Container',
                style: TextStyle(color: Colors.white),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
```

#### 解释
- **padding**: 内边距，组件内容与边框之间的距离。
- **margin**: 外边距，组件与其他组件之间的距离。
- **BoxDecoration**: 用于设置背景色、边框、阴影等效果。

### 3. **Row 和 Column**
- **Row**：用于水平排列子组件。
- **Column**：用于垂直排列子组件。

#### 示例代码
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
          title: Text('Row 和 Column 示例'),
        ),
        body: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                Container(
                  width: 100,
                  height: 100,
                  color: Colors.red,
                  child: Center(child: Text('红色')),
                ),
                Container(
                  width: 100,
                  height: 100,
                  color: Colors.green,
                  child: Center(child: Text('绿色')),
                ),
              ],
            ),
            SizedBox(height: 20), // Row 和 Column 之间的空白
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                Container(
                  width: 100,
                  height: 100,
                  color: Colors.blue,
                  child: Center(child: Text('蓝色')),
                ),
                Container(
                  width: 100,
                  height: 100,
                  color: Colors.yellow,
                  child: Center(child: Text('黄色')),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
```

#### 解释
- **Row** 和 **Column** 的 `mainAxisAlignment` 属性控制主轴方向的对齐方式，常见的值包括：
  - **MainAxisAlignment.start**：开始对齐。
  - **MainAxisAlignment.center**：居中对齐。
  - **MainAxisAlignment.end**：结束对齐。
  - **MainAxisAlignment.spaceAround**：子组件之间留有均匀的空白。
  
- **SizedBox**: 可以在两个组件之间添加空白距离。

### 4. **Stack**
`Stack` 组件允许子组件重叠显示，常用于创建复杂的层叠布局。

#### 示例代码
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
          title: Text('Stack 示例'),
        ),
        body: Stack(
          children: <Widget>[
            Container(
              width: 300,
              height: 300,
              color: Colors.blue,
            ),
            Positioned(
              top: 50,
              left: 50,
              child: Container(
                width: 200,
                height: 200,
                color: Colors.red,
              ),
            ),
            Positioned(
              top: 100,
              left: 100,
              child: Container(
                width: 100,
                height: 100,
                color: Colors.green,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
```

#### 解释
- **Stack**: 子组件会按顺序叠加在一起。
- **Positioned**: 用于在 `Stack` 中定位子组件，可以通过 `top`、`left`、`right` 和 `bottom` 属性来控制位置。

### 5. **Alignment（对齐方式）**
在 Flutter 中，`Alignment` 控制子组件在其父容器中的对齐方式。可以使用 `Align` 小部件或 `Container` 的 `alignment` 属性进行控制。

#### 示例代码
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
          title: Text('Align 示例'),
        ),
        body: Center(
          child: Container(
            width: 300,
            height: 300,
            color: Colors.blue,
            alignment: Alignment.bottomRight,
            child: Container(
              width: 100,
              height: 100,
              color: Colors.red,
            ),
          ),
        ),
      ),
    );
  }
}
```

#### 解释
- **Alignment**：`Alignment.bottomRight` 会将子组件对齐到右下角。
- 常见的对齐值有：`Alignment.center`、`Alignment.topLeft`、`Alignment.bottomRight` 等。

### 总结
- `Container` 是一个常用的布局组件，可以控制尺寸、内外边距、背景色等。
- `Row` 和 `Column` 用于水平和垂直布局。
- `Stack` 允许子组件层叠排列。
- `Align` 控制组件的对齐方式。

在学习这些布局组件后，你可以尝试构建更复杂的 UI 布局，并使用它们组合成实际应用场景中的界面设计。

这个是 Day 4 的完整教学内容，你可以通过动手实践这些例子，逐步熟悉 Flutter 的布局系统。

接下来我们进一步探讨一些在 Day 4 教学中未涉及的高级布局技巧和常见问题解决方案，让你更好地掌握 Flutter 布局系统。

### 6. **Flexible 和 Expanded**

在 `Row` 和 `Column` 布局中，常常需要根据可用空间来动态调整子组件的大小。Flutter 提供了两个非常有用的 Widget：**Flexible** 和 **Expanded**。

- **Flexible**：允许子组件在可用空间内根据其设定比例进行调整，但不会占据所有的可用空间。
- **Expanded**：与 `Flexible` 类似，但是它会强制子组件占据所有的可用空间。

#### 示例代码
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
          title: Text('Flexible 和 Expanded 示例'),
        ),
        body: Column(
          children: <Widget>[
            Row(
              children: [
                Expanded(
                  child: Container(
                    height: 100,
                    color: Colors.red,
                    child: Center(child: Text('Expanded 1')),
                  ),
                ),
                Expanded(
                  child: Container(
                    height: 100,
                    color: Colors.green,
                    child: Center(child: Text('Expanded 2')),
                  ),
                ),
              ],
            ),
            SizedBox(height: 20),
            Row(
              children: [
                Flexible(
                  flex: 1,
                  child: Container(
                    height: 100,
                    color: Colors.blue,
                    child: Center(child: Text('Flexible 1')),
                  ),
                ),
                Flexible(
                  flex: 2,
                  child: Container(
                    height: 100,
                    color: Colors.yellow,
                    child: Center(child: Text('Flexible 2')),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
```

#### 解释
- **Expanded**：红色和绿色的两个 `Container` 会均匀地分配 `Row` 的宽度，完全填充可用的水平空间。
- **Flexible**：蓝色和黄色的 `Container` 按 `flex` 属性的比例调整它们的宽度，`Flexible 2` 占据的宽度是 `Flexible 1` 的两倍。

### 7. **Wrap**

`Wrap` 是一个用于在行或列中排列子组件的布局，但当空间不足时，它会自动换行。与 `Row` 和 `Column` 的区别在于，`Wrap` 可以更好地适应内容较多的情况，避免组件溢出。

#### 示例代码
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
          title: Text('Wrap 示例'),
        ),
        body: Wrap(
          spacing: 8.0,
          runSpacing: 4.0,
          children: <Widget>[
            Chip(
              label: Text('标签 1'),
            ),
            Chip(
              label: Text('标签 2'),
            ),
            Chip(
              label: Text('标签 3'),
            ),
            Chip(
              label: Text('标签 4'),
            ),
            Chip(
              label: Text('标签 5'),
            ),
            Chip(
              label: Text('标签 6'),
            ),
            Chip(
              label: Text('标签 7'),
            ),
          ],
        ),
      ),
    );
  }
}
```

#### 解释
- **spacing**: 每个组件之间的水平间距。
- **runSpacing**: 每行之间的垂直间距。
- **Wrap**: 当子组件超过一行的可用空间时，会自动换到下一行。

### 8. **GridView**

`GridView` 是一个用于在网格中显示子组件的布局组件，通常用于显示图片、卡片等内容。`GridView` 提供了多个构造函数，其中最常用的是 `GridView.count` 和 `GridView.builder`。

#### 示例代码
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
          title: Text('GridView 示例'),
        ),
        body: GridView.count(
          crossAxisCount: 2, // 每行显示2个子组件
          crossAxisSpacing: 10,
          mainAxisSpacing: 10,
          padding: EdgeInsets.all(10),
          children: List.generate(6, (index) {
            return Container(
              color: Colors.blue,
              child: Center(
                child: Text(
                  'Item $index',
                  style: TextStyle(color: Colors.white),
                ),
              ),
            );
          }),
        ),
      ),
    );
  }
}
```

#### 解释
- **crossAxisCount**: 控制每行显示的子组件数量。
- **crossAxisSpacing**: 子组件之间的水平间距。
- **mainAxisSpacing**: 子组件之间的垂直间距。

### 9. **ListView**

`ListView` 是一个非常常用的布局组件，用于垂直显示多个组件。它适用于创建滚动列表，支持大量数据展示。

#### 示例代码
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
          title: Text('ListView 示例'),
        ),
        body: ListView.builder(
          itemCount: 10,
          itemBuilder: (context, index) {
            return ListTile(
              title: Text('Item $index'),
              leading: Icon(Icons.star),
              trailing: Icon(Icons.arrow_forward),
              onTap: () {
                print('点击了Item $index');
              },
            );
          },
        ),
      ),
    );
  }
}
```

#### 解释
- **ListView.builder**: 根据 `itemCount` 创建一个动态的列表。使用 `itemBuilder` 返回每一行的组件。
- **ListTile**: Flutter 提供的一个列表项组件，用于快速创建列表中的每一行。

### 10. **总结与练习建议**

在 Day 4 的学习中，你了解了如何使用 Flutter 中的布局组件来构建复杂的 UI。这里提供了一些建议来巩固你今天的学习：
- 创建一个登录页面，使用 `Column` 布局，包含两个输入框和一个按钮。
- 使用 `GridView` 显示图片网格，每个网格项显示一张图片和标题。
- 使用 `Wrap` 创建一个动态标签系统，允许用户点击选择不同标签。

通过不断练习和调整这些布局，你将会更熟悉 Flutter 的布局机制，从而能够设计出更具表现力的用户界面。