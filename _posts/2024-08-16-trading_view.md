---
layout: post
title:  trading view basic script
date:   2024-08-16 12:24:29 +0800
categories: quant
---

```py
// This Pine Script™ code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
//@version=5
indicator("EMA Example", overlay=true)

// 参数设置
length = input(14, title="EMA Length")
source = close  // 使用收盘价计算

// 计算 EMA
ema_shot = ta.ema(source, length)
ema_long = ta.ema(close,25)

// 绘制 EMA
plot(ema_shot, title="EMA", color=color.blue, linewidth=2)
```