---
layout: post
title:  go 结构体
date:   2025-02-26 11:24:29 +0800
categories: 
    - go
---

```go
package main

import "fmt"

// 定义接口
type Speaker interface {
	Speak() string
}

// 结构体实现接口
type Dog struct{}

func (d Dog) Speak() string {
	return "汪汪!"
}

type Cat struct{}

func (c Cat) Speak() string {
	return "喵喵!"
}

func main() {
	var s Speaker
	s = Dog{}
	fmt.Println("狗:", s.Speak())

	s = Cat{}
	fmt.Println("猫:", s.Speak())
}
```