import pandas as pd
import numpy as np
import talib
import joblib
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, make_scorer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import time

# 尝试导入 seaborn（可选）
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    pass

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# =====================
# 参数
# =====================
RSI = 14
EMA = 200
ATR = 14
MIN_BARS = 210
FUTURE = 5

TRAIN_WINDOW = 1000
TEST_WINDOW = 200

# =====================
# 网格搜索配置
# =====================
USE_GRID_SEARCH = True  # 是否使用网格搜索
GRID_SEARCH_MODE = "first_only"  # "first_only": 只对第一个模型搜索, "all": 对所有模型搜索, "skip": 跳过网格搜索
CV_FOLDS = 3  # 交叉验证折数（仅用于网格搜索）

# 定义参数网格
param_grid = {
    'n_estimators': [500, 800, 1000],
    'max_depth': [10, 15, 20],
    'learning_rate': [0.001, 0.003, 0.005],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

# 默认参数（如果跳过网格搜索或作为初始值）
default_params = {
    'n_estimators': 1000,
    'max_depth': 20,
    'learning_rate': 0.001,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'random_state': 42
}

df = pd.read_csv(
    "XAUUSD.csv",
    header=None,  # 因为你 CSV 没有列名
    names=["date", "time", "open", "high", "low", "close", "volume"]
)

df.columns = [c.lower() for c in df.columns]

cl = df["close"].values
hi = df["high"].values
lo = df["low"].values

df["rsi"] = talib.RSI(cl, RSI)
df["ema200"] = talib.EMA(cl, EMA)
df["atr"] = talib.ATR(hi, lo, cl, ATR)
df["ret_5"] = (df["close"] - df["close"].shift(5)) / df["close"].shift(5)
df["vol"] = df["close"].rolling(10).std()
df['tema'] = talib.TEMA(cl, EMA)
df['tema_dist'] = (df['close'] - df['tema']) / df['tema']

df["ema_dist"] = (df["close"] - df["ema200"]) / df["ema200"]

df["y"] = (df["close"].shift(-FUTURE) > df["close"]).astype(int)

df = df.iloc[MIN_BARS:-FUTURE].dropna()

FEATURES = ["rsi", "atr", "ret_5", "vol", "ema_dist"]

models = []
train_accuracies = []
test_accuracies = []
train_precisions = []
test_precisions = []
train_recalls = []
test_recalls = []
train_f1s = []
test_f1s = []
model_indices = []
best_params_history = []  # 记录每个模型的最佳参数

print("=" * 60)
print("开始 Walk-Forward 训练...")
if USE_GRID_SEARCH and GRID_SEARCH_MODE != "skip":
    print(f"网格搜索模式: {GRID_SEARCH_MODE}")
    print(f"参数网格大小: {np.prod([len(v) for v in param_grid.values()])} 种组合")
print("=" * 60)

# 最佳参数（将在第一个模型训练后确定）
best_params = default_params.copy()

start = 0
model_idx = 0
while start + TRAIN_WINDOW + TEST_WINDOW < len(df):
    train = df.iloc[start:start+TRAIN_WINDOW]
    test = df.iloc[start+TRAIN_WINDOW:start+TRAIN_WINDOW+TEST_WINDOW]

    # 决定是否进行网格搜索
    should_grid_search = (
        USE_GRID_SEARCH and 
        GRID_SEARCH_MODE != "skip" and 
        (GRID_SEARCH_MODE == "all" or (GRID_SEARCH_MODE == "first_only" and model_idx == 0))
    )
    
    if should_grid_search:
        print(f"\n模型 {model_idx + 1}: 正在进行网格搜索...")
        start_time = time.time()
        
        # 使用 F1 分数作为评分标准
        f1_scorer = make_scorer(f1_score, zero_division=0)
        
        # 创建基础模型
        base_model = XGBClassifier(
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1  # 使用所有CPU核心
        )
        
        # 时间序列交叉验证（保持时间顺序）
        tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
        
        # 执行网格搜索
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring=f1_scorer,
            cv=tscv,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(train[FEATURES], train["y"])
        
        # 获取最佳参数
        best_params = {
            **grid_search.best_params_,
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'random_state': 42
        }
        
        elapsed_time = time.time() - start_time
        print(f"网格搜索完成 (耗时: {elapsed_time:.2f}秒)")
        print(f"最佳参数: {best_params}")
        print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")
        
        # 使用最佳参数创建模型
        model = XGBClassifier(**best_params)
        model.fit(train[FEATURES], train["y"])
        
    else:
        # 使用之前找到的最佳参数或默认参数
        if model_idx == 0 and not USE_GRID_SEARCH:
            print(f"模型 {model_idx + 1}: 使用默认参数")
        else:
            print(f"模型 {model_idx + 1}: 使用参数 {best_params.get('n_estimators')} trees, "
                  f"depth={best_params.get('max_depth')}, lr={best_params.get('learning_rate')}")
        
        model = XGBClassifier(**best_params)
        model.fit(train[FEATURES], train["y"])
    
    models.append(model)
    best_params_history.append(best_params.copy())
    
    # 预测
    train_pred = model.predict(train[FEATURES])
    test_pred = model.predict(test[FEATURES])
    
    # 计算指标
    train_acc = accuracy_score(train["y"], train_pred)
    test_acc = accuracy_score(test["y"], test_pred)
    train_prec = precision_score(train["y"], train_pred, zero_division=0)
    test_prec = precision_score(test["y"], test_pred, zero_division=0)
    train_rec = recall_score(train["y"], train_pred, zero_division=0)
    test_rec = recall_score(test["y"], test_pred, zero_division=0)
    train_f1 = f1_score(train["y"], train_pred, zero_division=0)
    test_f1 = f1_score(test["y"], test_pred, zero_division=0)
    
    # 保存指标
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    train_precisions.append(train_prec)
    test_precisions.append(test_prec)
    train_recalls.append(train_rec)
    test_recalls.append(test_rec)
    train_f1s.append(train_f1)
    test_f1s.append(test_f1)
    model_indices.append(model_idx)
    
    # 打印进度
    print(f"模型 {model_idx + 1}: 训练集准确率={train_acc:.4f}, 测试集准确率={test_acc:.4f}, "
          f"测试集F1={test_f1:.4f}")
    
    start += TEST_WINDOW
    model_idx += 1

joblib.dump(models, "xgb_walkforward_models.pkl")
print(f"\n已保存 {len(models)} 个 walk-forward 模型")

# =====================
# 可视化
# =====================
print("\n生成可视化图表...")

fig = plt.figure(figsize=(16, 12))

# 1. 准确率对比
plt.subplot(3, 2, 1)
plt.plot(model_indices, train_accuracies, 'o-', label='训练集准确率', linewidth=2, markersize=6)
plt.plot(model_indices, test_accuracies, 's-', label='测试集准确率', linewidth=2, markersize=6)
plt.xlabel('模型编号', fontsize=12)
plt.ylabel('准确率', fontsize=12)
plt.title('训练集 vs 测试集准确率', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])

# 2. 精确率对比
plt.subplot(3, 2, 2)
plt.plot(model_indices, train_precisions, 'o-', label='训练集精确率', linewidth=2, markersize=6)
plt.plot(model_indices, test_precisions, 's-', label='测试集精确率', linewidth=2, markersize=6)
plt.xlabel('模型编号', fontsize=12)
plt.ylabel('精确率', fontsize=12)
plt.title('训练集 vs 测试集精确率', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])

# 3. 召回率对比
plt.subplot(3, 2, 3)
plt.plot(model_indices, train_recalls, 'o-', label='训练集召回率', linewidth=2, markersize=6)
plt.plot(model_indices, test_recalls, 's-', label='测试集召回率', linewidth=2, markersize=6)
plt.xlabel('模型编号', fontsize=12)
plt.ylabel('召回率', fontsize=12)
plt.title('训练集 vs 测试集召回率', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])

# 4. F1分数对比
plt.subplot(3, 2, 4)
plt.plot(model_indices, train_f1s, 'o-', label='训练集F1', linewidth=2, markersize=6)
plt.plot(model_indices, test_f1s, 's-', label='测试集F1', linewidth=2, markersize=6)
plt.xlabel('模型编号', fontsize=12)
plt.ylabel('F1分数', fontsize=12)
plt.title('训练集 vs 测试集F1分数', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])

# 5. 特征重要性（使用最后一个模型）
plt.subplot(3, 2, 5)
feature_importance = models[-1].feature_importances_
feature_df = pd.DataFrame({
    'feature': FEATURES,
    'importance': feature_importance
}).sort_values('importance', ascending=True)
plt.barh(feature_df['feature'], feature_df['importance'], color='steelblue')
plt.xlabel('重要性', fontsize=12)
plt.ylabel('特征', fontsize=12)
plt.title('特征重要性（最后一个模型）', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')

# 6. 所有模型的平均特征重要性
plt.subplot(3, 2, 6)
avg_importance = np.mean([m.feature_importances_ for m in models], axis=0)
avg_feature_df = pd.DataFrame({
    'feature': FEATURES,
    'importance': avg_importance
}).sort_values('importance', ascending=True)
plt.barh(avg_feature_df['feature'], avg_feature_df['importance'], color='coral')
plt.xlabel('平均重要性', fontsize=12)
plt.ylabel('特征', fontsize=12)
plt.title('所有模型的平均特征重要性', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('training_visualization.png', dpi=300, bbox_inches='tight')
print("可视化图表已保存为: training_visualization.png")
plt.show()

# =====================
# 参数变化可视化（如果使用网格搜索）
# =====================
if USE_GRID_SEARCH and GRID_SEARCH_MODE == "all" and len(best_params_history) > 0:
    print("\n生成参数变化可视化图表...")
    
    # 提取参数值
    n_estimators_list = [p.get('n_estimators', 0) for p in best_params_history]
    max_depth_list = [p.get('max_depth', 0) for p in best_params_history]
    lr_list = [p.get('learning_rate', 0) for p in best_params_history]
    subsample_list = [p.get('subsample', 0) for p in best_params_history]
    colsample_list = [p.get('colsample_bytree', 0) for p in best_params_history]
    
    fig_params = plt.figure(figsize=(16, 10))
    
    # 1. n_estimators
    plt.subplot(2, 3, 1)
    plt.plot(model_indices, n_estimators_list, 'o-', color='steelblue', linewidth=2, markersize=6)
    plt.xlabel('模型编号', fontsize=12)
    plt.ylabel('n_estimators', fontsize=12)
    plt.title('树的数量变化', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 2. max_depth
    plt.subplot(2, 3, 2)
    plt.plot(model_indices, max_depth_list, 's-', color='coral', linewidth=2, markersize=6)
    plt.xlabel('模型编号', fontsize=12)
    plt.ylabel('max_depth', fontsize=12)
    plt.title('最大深度变化', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 3. learning_rate
    plt.subplot(2, 3, 3)
    plt.plot(model_indices, lr_list, '^-', color='green', linewidth=2, markersize=6)
    plt.xlabel('模型编号', fontsize=12)
    plt.ylabel('learning_rate', fontsize=12)
    plt.title('学习率变化', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 4. subsample
    plt.subplot(2, 3, 4)
    plt.plot(model_indices, subsample_list, 'd-', color='purple', linewidth=2, markersize=6)
    plt.xlabel('模型编号', fontsize=12)
    plt.ylabel('subsample', fontsize=12)
    plt.title('子样本比例变化', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 5. colsample_bytree
    plt.subplot(2, 3, 5)
    plt.plot(model_indices, colsample_list, 'v-', color='orange', linewidth=2, markersize=6)
    plt.xlabel('模型编号', fontsize=12)
    plt.ylabel('colsample_bytree', fontsize=12)
    plt.title('列采样比例变化', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 6. 参数与性能的关系
    plt.subplot(2, 3, 6)
    scatter = plt.scatter(n_estimators_list, test_f1s, c=max_depth_list, 
                         s=100, alpha=0.6, cmap='viridis')
    plt.xlabel('n_estimators', fontsize=12)
    plt.ylabel('测试集F1分数', fontsize=12)
    plt.title('参数与性能关系', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='max_depth')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('parameters_visualization.png', dpi=300, bbox_inches='tight')
    print("参数可视化图表已保存为: parameters_visualization.png")
    plt.show()

# =====================
# 打印统计摘要
# =====================
print("\n" + "=" * 60)
print("训练统计摘要")
print("=" * 60)
print(f"总模型数: {len(models)}")
if USE_GRID_SEARCH and GRID_SEARCH_MODE != "skip":
    print(f"网格搜索模式: {GRID_SEARCH_MODE}")
    if GRID_SEARCH_MODE == "first_only":
        print(f"使用的参数: {best_params_history[0]}")
print(f"\n测试集性能统计:")
print(f"  平均准确率: {np.mean(test_accuracies):.4f} ± {np.std(test_accuracies):.4f}")
print(f"  平均精确率: {np.mean(test_precisions):.4f} ± {np.std(test_precisions):.4f}")
print(f"  平均召回率: {np.mean(test_recalls):.4f} ± {np.std(test_recalls):.4f}")
print(f"  平均F1分数: {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}")
print(f"\n最佳模型 (按测试集F1):")
best_idx = np.argmax(test_f1s)
print(f"  模型编号: {best_idx}")
print(f"  测试集准确率: {test_accuracies[best_idx]:.4f}")
print(f"  测试集F1: {test_f1s[best_idx]:.4f}")
if USE_GRID_SEARCH and GRID_SEARCH_MODE == "all":
    print(f"  使用的参数: {best_params_history[best_idx]}")
print("=" * 60)

# 保存最佳参数到文件
if USE_GRID_SEARCH and GRID_SEARCH_MODE != "skip":
    params_df = pd.DataFrame(best_params_history)
    params_df.to_csv('best_params_history.csv', index=False)
    print(f"\n参数历史已保存到: best_params_history.csv")
