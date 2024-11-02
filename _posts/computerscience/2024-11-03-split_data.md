---
layout: post
title : split data
date : 2024-11-03 11:24:29 +0800
categories: 
    - stats
---

### 使用 train_test_split 多次划分数据

```py
from sklearn.model_selection import train_test_split

# 假设有数据 X 和标签 y
X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
y = [0, 1, 0, 1, 0]

# 第一次划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("第一次划分:")
print("X_train:", X_train)
print("X_test:", X_test)

# 第二次划分，设置不同的 random_state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print("\n第二次划分:")
print("X_train:", X_train)
print("X_test:", X_test)
```

### 使用 KFold 进行交叉验证划分

KFold 将数据分成 k 份，用于交叉验证

```py
from sklearn.model_selection import KFold
import numpy as np

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y = np.array([0, 1, 0, 1, 0])

kf = KFold(n_splits=3)  # 将数据分成3份
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("Train Index:", train_index, "Test Index:", test_index)
    print("X_train:", X_train, "X_test:", X_test)
```

### 使用 StratifiedKFold 保证分布一致的交叉验证划分

StratifiedKFold 保证每个划分中的类别分布一致。

```py
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=3)
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("Train Index:", train_index, "Test Index:", test_index)
    print("X_train:", X_train, "X_test:", X_test)
```

### 使用时间序列划分 (TimeSeriesSplit)

在时间序列数据上，使用普通的 KFold 划分会导致未来的数据泄漏。TimeSeriesSplit 是一种适用于时间序列数据的交叉验证方法，通过滚动划分方式来保持数据顺序。

```py
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

# 假设有时间序列数据 X
X = np.arange(10)

tscv = TimeSeriesSplit(n_splits=3)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    print("Train:", X_train, "Test:", X_test)
```

### 使用留一法交叉验证 (LeaveOneOut)

LeaveOneOut 将数据集中的每一个样本都单独当作测试集，其他的样本用作训练集。这适合数据集非常小的情况，但计算量较大。

```py
from sklearn.model_selection import LeaveOneOut
import numpy as np

X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 0])

loo = LeaveOneOut()
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("Train:", X_train, "Test:", X_test)
```

### 使用留P法交叉验证 (LeavePOut)

LeavePOut 是 LeaveOneOut 的扩展，将 P 个样本留作测试集，其他样本作为训练集。P 越大，划分组合越多，因此计算量较大。

```py
from sklearn.model_selection import LeavePOut

X = np.array([[1, 2], [3, 4], [5, 6]])
lpo = LeavePOut(p=2)
for train_index, test_index in lpo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    print("Train:", X_train, "Test:", X_test)
```

### 使用分层洗牌划分 (StratifiedShuffleSplit)

StratifiedShuffleSplit 是 StratifiedKFold 和 ShuffleSplit 的组合，它通过分层抽样保证训练集和测试集中类别分布一致，适合分类问题中数据分布不均的情况。

```py
from sklearn.model_selection import StratifiedShuffleSplit

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y = np.array([0, 0, 1, 1, 1])

sss = StratifiedShuffleSplit(n_splits=3, test_size=0.4, random_state=42)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("Train:", X_train, "Test:", X_test)
```

### 使用Bootstrap 采样 (Bootstrap)

Bootstrap 方法通过重复采样的方式创建多个训练集，常用于估计模型在不同数据样本上的稳定性

```py
from sklearn.utils import resample

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y = np.array([0, 0, 1, 1, 1])

# 使用随机重采样生成 Bootstrap 数据
for i in range(3):  # 重复生成3个不同的样本
    X_resample, y_resample = resample(X, y, n_samples=4, random_state=i)
    print("Bootstrap Sample", i+1)
    print("X_resample:", X_resample)
    print("y_resample:", y_resample)
```

### 使用随机分割交叉验证 (ShuffleSplit)

ShuffleSplit 适用于数据较多时的交叉验证，通过随机抽样多次生成训练集和测试集，以随机化的方式覆盖数据集。

```py
from sklearn.model_selection import ShuffleSplit

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y = np.array([0, 0, 1, 1, 1])

ss = ShuffleSplit(n_splits=3, test_size=0.4, random_state=42)
for train_index, test_index in ss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("Train:", X_train, "Test:", X_test)
```

### 使用分层时间序列划分 (StratifiedGroupKFold)

对于包含群组或分层特性的时间序列数据，可以使用 `StratifiedGroupKFold` 进行分层组划分，这样可以在不打乱时间顺序的情况下划分数据集。

```py
from sklearn.model_selection import StratifiedGroupKFold

# 示例数据集，包含样本的特征、类别标签和分组信息
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y = np.array([0, 0, 1, 1, 1])
groups = np.array([1, 1, 2, 2, 3])  # 假设有分组特性

sgkf = StratifiedGroupKFold(n_splits=3)
for train_index, test_index in sgkf.split(X, y, groups=groups):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("Train:", X_train, "Test:", X_test)
```