package main

import "fmt"

// 定义结构体
type User struct {
	Name string
	Age  int
}

// 给结构体绑定方法
func (u User) Greet() {
	fmt.Printf("你好，我是 %s，今年 %d 岁。\n", u.Name, u.Age)
}

func main() {
	user := User{Name: "张三", Age: 30}
	user.Greet()
}
