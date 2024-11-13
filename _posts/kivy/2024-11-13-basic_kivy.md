---
layout: post
title:  kivy 基础
date:   2024-11-13 11:24:29 +0800
categories: 
    - python 
    - kivy
---

安装 Kivy 和创建基本的 "I\'m geo" 应用

```py
from kivy.app import App
from kivy.uix.label import Label

class app(App):
    def build(self):
        return Label(text="I\'m geo")

if __name__ == "__main__":
    app().run()
```

### label

对于字体的颜色和其他相关的设定

```py
Label:
    text: "I\'m geo"
    font_size: '16sp'
    opacity : 0.5
    color : (1,1,1,1)
```

```py
from kivy.app import App
from kivy.uix.label import Label

class app(App):
    ...

if __name__ == '__main__':
    app().run()
```

或者直接设定如下 : 

```py
from kivy.app import App
from kivy.uix.label import Label

class app(App):
    def build(self):
        return Label(text='Hello, Kivy!', font_size='24sp', color=(1, 0, 0, 1))  # 红色文本

if __name__ == '__main__':
    app().run()
```

### button

```py
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

class MyWidget(BoxLayout):
    def on_button_click(self):
        print("Button clicked!")

class MyApp(App):
    def build(self):
        return MyWidget()

if __name__ == '__main__':
    MyApp().run()

```

```py
<MyWidget>:
    orientation: 'vertical'
    
    Label:
        text: 'Hello from KV!'
        font_size: '24sp'
        color: (1, 0, 0, 1)  # 红色文本

    Button:
        text: 'Click Me'
        font_size: '20sp'
        on_press: root.on_button_click()

```

直接嵌入完整的设定

```py
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout

class MyApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')  # 垂直布局
        label = Label(text="Welcome to Kivy!", font_size='24sp')
        button = Button(text="Click Me", font_size='20sp')
        
        button.bind(on_press=self.on_button_click)
        layout.add_widget(label)
        layout.add_widget(button)
        
        return layout

    def on_button_click(self, instance):
        print("Button clicked!")

if __name__ == '__main__':
    MyApp().run()
```