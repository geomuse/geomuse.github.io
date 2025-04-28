---
layout: post
title:  go 变数申明
date:   2025-02-24 11:24:29 +0800
categories: 
    - go
---

- 熟悉 Go 语言语法（变量、函数、结构体、接口等）
- 理解 Go 并发编程（goroutines、channels）
- 掌握 Go 模块管理（go mod）

```go
package main

import "fmt"

func main() {
	// 变量声明
	var name string = "Go"
	age := 10 // 类型推导
	var pi float64 = 3.1415

	// 数组和切片
	numbers := []int{1, 2, 3, 4, 5}

	// 映射
	user := map[string]string{"name": "Alice", "age": "25"}

	// 输出
	fmt.Println("Hello,", name)
	fmt.Println("Pi 值:", pi)
	fmt.Println("年龄:", age)
	fmt.Println("数字列表:", numbers)
	fmt.Println("用户:", user)
}
```

```bash
Hello, Go
Pi 值: 3.1415
年龄: 10
数字列表: [1 2 3 4 5]
```