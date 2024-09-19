---
layout: post
title:  react 基本
date:   2024-09-20 11:24:29 +0800
categories:
    - react
---

### React 基本教学

React 是由 Facebook 开发的一个用于构建用户界面的开源 JavaScript 库。它主要用于构建单页应用，可以高效地更新和渲染数据变化的用户界面。

#### 一、React 是什么？

- **声明式**：React 使创建交互式用户界面变得轻而易举。为应用程序的每个状态设计简单的视图，当数据变化时，React 能高效地更新和渲染正确的组件。
- **组件化**：构建封装了自身状态的组件，然后将它们组合起来形成复杂的用户界面。

#### 二、核心概念

##### 1. 组件

组件是 React 的核心。一个组件可以是一个函数或一个类，它接收输入（称为 "props"），并返回要在屏幕上显示的内容（React 元素）。

##### 2. JSX

JSX 是一种 JavaScript 的语法扩展，建议在 React 中使用。JSX 看起来像是模板语言，但它具有 JavaScript 的全部功能。

```jsx
function HelloWorld() {
  return <h1>Hello, World!</h1>;
}
```

##### 3. 状态（State）和属性（Props）

- **Props**：组件的输入参数，使用属性的方式从父组件传递到子组件，组件内部无法修改。
- **State**：组件内部可变的数据源，用于存储组件内部的状态信息。

##### 4. 生命周期

React 组件有一系列生命周期方法，可以在组件创建、更新和销毁时执行特定的操作。

#### 四、详细讲解

##### 1. 创建组件

组件可以是函数式或类式的。

**函数式组件：**

```jsx
function Welcome(props) {
  return <h1>Hello, {props.name}</h1>;
}
```

**类组件：**

```jsx
class Welcome extends React.Component {
  render() {
    return <h1>Hello, {this.props.name}</h1>;
  }
}
```

##### 2. 使用 JSX

JSX 允许我们在 JavaScript 中直接写 HTML 结构。

```jsx
const element = <h1>Hello, world!</h1>;
```

##### 3. 管理状态和属性

**使用状态（State）：**

```jsx
class Clock extends React.Component {
  constructor(props) {
    super(props);
    this.state = {date: new Date()};
  }

  render() {
    return (
      <div>
        <h1>现在的时间是 {this.state.date.toLocaleTimeString()}。</h1>
      </div>
    );
  }
}
```

**更新状态：**

```jsx
this.setState({date: new Date()});
```

##### 4. 事件处理

React 的事件处理与 DOM 元素的事件处理类似，但是有一些语法差异。

```jsx
<button onClick={this.handleClick}>点击我</button>
```

事件处理器可以绑定在组件的方法中：

```jsx
handleClick() {
  console.log('按钮被点击了');
}
```

##### 5. 组件生命周期方法

常用的生命周期方法有：

- `componentDidMount()`：组件已经被渲染到 DOM 中后运行。
- `componentDidUpdate(prevProps, prevState)`：组件更新后运行。
- `componentWillUnmount()`：组件从 DOM 中移除之前运行。

示例：

```jsx
class Clock extends React.Component {
  constructor(props) {
    super(props);
    this.state = {date: new Date()};
  }

  componentDidMount() {
    this.timerID = setInterval(
      () => this.tick(),
      1000
    );
  }

  componentWillUnmount() {
    clearInterval(this.timerID);
  }

  tick() {
    this.setState({
      date: new Date()
    });
  }

  render() {
    return (
      <h2>现在是 {this.state.date.toLocaleTimeString()}.</h2>
    );
  }
}
```

