from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 数据
X, y = load_iris(return_X_y=True)

# 切训练 / 测试
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=200))
])

# 训练
pipe.fit(X_train, y_train)

# 预测
print(pipe.predict(X_test))
print(pipe.score(X_test, y_test))
