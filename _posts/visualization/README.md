好，这个我给你一套**「30 天 · Python 数据视觉化实战教学路线」**，走的是**从零 → 能做专业图表 → 能讲故事 → 能做作品集**的路线，偏**实用 + 可复现**，不是只画漂亮图那种 👌

默认你**会一点 Python 基础**（变量、列表、函数），不行也能硬跟。

---

# 🧭 总体路线设计

* **第 1–10 天**：基础与核心库（Matplotlib / Pandas）
* **第 11–20 天**：进阶图表 + Seaborn + 统计可视化
* **第 21–26 天**：交互式可视化（Plotly / Dash）
* **第 27–30 天**：数据故事 + 项目 + 作品集

---

# 📅 第 1 阶段：可视化基础（Day 1–10）

## Day 1｜数据视觉化在干嘛？

* 什么是数据视觉化
* 好图 vs 坏图
* Python 可视化生态全图

  * Matplotlib / Seaborn / Plotly / Altair
* 安装环境

```bash
pip install matplotlib pandas seaborn plotly
```

---

## Day 2｜Matplotlib 入门

* `plt.plot()`
* 线图（Line Chart）
* x / y

```python
import matplotlib.pyplot as plt

plt.plot([1,2,3],[4,6,5])
plt.show()
```

---

## Day 3｜图表元素控制

* 标题、标签、图例
* 颜色、线型、marker

```python
plt.plot(x, y, label="price")
plt.legend()
```

---

## Day 4｜常见基础图表

* 折线图
* 柱状图
* 饼图
* 直方图

---

## Day 5｜Pandas + 可视化

* `DataFrame.plot()`
* CSV → 图表

```python
df.plot(kind="line")
```

---

## Day 6｜时间序列可视化（🔥重点）

* 日期索引
* 金融 / 流量 / 传感器数据

```python
df['date'] = pd.to_datetime(df['date'])
df.set_index('date').plot()
```

---

## Day 7｜多图组合

* subplot
* 一张图多个子图

```python
plt.subplot(2,1,1)
```

---

## Day 8｜图表美化基础

* 字体
* 中文显示
* DPI / figsize

```python
plt.figure(figsize=(10,5), dpi=120)
```

---

## Day 9｜常见错误 & 调试

* 图表不显示
* 中文乱码
* 轴比例错误

---

## Day 10｜小项目 ①

🎯 **项目**：

> 用 CSV 数据画「某商品价格趋势图 + 成交量柱状图」

---

# 📅 第 2 阶段：统计 & 美学（Day 11–20）

## Day 11｜Seaborn 入门

* 为什么用 Seaborn
* 自动美化

```python
import seaborn as sns
sns.lineplot(data=df)
```

---

## Day 12｜分类数据可视化

* barplot
* countplot
* hue（分组对比）

---

## Day 13｜分布分析（🔥非常重要）

* histogram
* KDE
* boxplot
* violin plot

---

## Day 14｜关系分析

* scatter
* regression plot
* pairplot

---

## Day 15｜热力图（Heatmap）

* correlation matrix

```python
sns.heatmap(df.corr(), annot=True)
```

---

## Day 16｜统计思维 + 可视化

* 均值 vs 中位数
* 离群值
* 偏态分布

---

## Day 17｜配色理论

* Sequential / Diverging / Categorical
* 调色盘选择

---

## Day 18｜信息密度 & 可读性

* 避免视觉误导
* 轴截断问题
* 数据比例原则

---

## Day 19｜图表用于「解释」而不是「炫技」

* 选择正确图表
* 一图一重点

---

## Day 20｜小项目 ②

🎯 **项目**：

> 分析一份用户数据，输出
>
> * 年龄分布
> * 收入 vs 消费关系
> * 分类对比图

---

# 📅 第 3 阶段：交互式可视化（Day 21–26）

## Day 21｜Plotly 入门（🔥）

* 交互式 hover

```python
import plotly.express as px
px.line(df, x="date", y="price")
```

---

## Day 22｜交互式金融图表

* K 线图
* 滑动缩放
* tooltip

---

## Day 23｜动态图表

* animation_frame
* 时间变化展示

---

## Day 24｜Dashboard 思维

* 多图联动
* Filter / Dropdown

---

## Day 25｜Dash 入门

* Python 写网页图表
* 简单 Dashboard

---

## Day 26｜部署概念

* 本地运行
* 作品展示思路（GitHub / Streamlit / Dash）

---

# 📅 第 4 阶段：数据故事 & 作品集（Day 27–30）

## Day 27｜数据叙事（Data Storytelling）

* 背景
* 冲突
* 结论
* 行动建议

---

## Day 28｜真实数据源

* Kaggle
* 政府数据
* 金融数据
* API 数据

---

## Day 29｜最终项目

🎯 **任选一个**：

* 股价分析 Dashboard
* 电商销售分析
* 用户行为分析
* ESG / 财务指标可视化（很适合你）

---

## Day 30｜作品集 & 进阶方向

* 项目结构
* README 怎么写
* 进阶路线：

  * 商业分析
  * 金融可视化
  * 学术可视化
  * Dashboard 工程师

---

# 🎁 我可以继续帮你做的

如果你愿意，我可以：

* ✅ **每天一讲 + 示例代码**
* ✅ **给你真实数据集**
* ✅ **偏金融 / 量化 / 商业分析方向定制**
* ✅ **做成“讲师级讲稿”或“学生练习版”**

你直接跟我说一句就行👇
👉 **「我要 Day 1～3 详细讲解」**
👉 **「偏金融数据视觉化」**
👉 **「我要能拿去教学用的版本」**
