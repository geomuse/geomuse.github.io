---
layout: post
title : data cleaning
date : 2024-11-08 11:24:29 +0800
categories: 
    - stats
---

### 初步了解数据

```py
# 查看数据的前5行
print(df.head())

# 查看数据的基本信息
print(df.info())

# 查看数据的统计描述
print(df.describe())

# 查看数据的形状（行数和列数）
print(df.shape)
```

### 处理缺失值

```py
# 检查每列的缺失值数量
print(df.isnull().sum())

# 删除包含缺失值的行
df.dropna(inplace=True)

# 删除全部为缺失值的列
df.dropna(axis=1, how='all', inplace=True)
```

### 填充缺失值

```py
# 用特定值填充
df['column_name'].fillna(0, inplace=True)

# 用均值填充
df['column_name'].fillna(df['column_name'].mean(), inplace=True)

# 用中位数填充
df['column_name'].fillna(df['column_name'].median(), inplace=True)

# 用众数填充
df['column_name'].fillna(df['column_name'].mode()[0], inplace=True)
```

### 处理重复数据

```py
# 检查重复行的数量
print(df.duplicated().sum())
# 删除重复行
df.drop_duplicates(inplace=True)
```

### 转换数据类型

```py
# 转换为整数
df['column_name'] = df['column_name'].astype(int)

# 转换为浮点数
df['column_name'] = df['column_name'].astype(float)

# 转换为字符串
df['column_name'] = df['column_name'].astype(str)

# 转换为日期时间格式
df['date_column'] = pd.to_datetime(df['date_column'])
```

### 特征工程

创建新的特征或转换现有特征，以提高模型性能。

```py
# 创建一个新特征，表示价格与面积的比值
df['price_per_sqft'] = df['price'] / df['area']
```

```py
# 对数转换，处理右偏分布
df['log_price'] = np.log(df['price'])
```

### 数据标准化与归一化

```py
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df['standardized_column'] = scaler.fit_transform(df[['column_name']])
```

```py
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df['normalized_column'] = scaler.fit_transform(df[['column_name']])
```

### 编码分类变量

大多数机器学习算法（如线性回归、神经网络、SVM等）只能处理数值型数据，无法直接处理文本或字符串形式的分类变量。因此，需要将这些分类变量转换为数值形式，即对其进行编码。

#### 独热编码（One-Hot Encoding）

适用于无序的分类变量，类别数量较少时效果较好。

```py
# 使用pandas的get_dummies方法
df = pd.get_dummies(df, columns=['categorical_column'])
```

####  标签编码（Label Encoding）

适用于有序的分类变量，如教育程度、等级等。

```py
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['encoded_column'] = le.fit_transform(df['categorical_column'])
```

### 二值编码（Binary Encoding）

适用于高基数（类别数量多）的分类变量，能减少维度。

```py
import category_encoders as ce

encoder = ce.BinaryEncoder(cols=['color'])
df = encoder.fit_transform(df)
```

### 频率编码（Frequency Encoding）

当类别的频率与目标变量相关时，可以考虑使用。

```py
freq_encoding = df['color'].value_counts() / len(df)
df['color_encoded'] = df['color'].map(freq_encoding)
```

### 目标编码（Target Encoding）

适用于高基数分类变量，但需注意过拟合风险。

```py
mean_encoding = df.groupby('color')['price'].mean()
df['color_encoded'] = df['color'].map(mean_encoding)
```