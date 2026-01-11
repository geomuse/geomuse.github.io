import pandas as pd
import numpy as np
import talib
import joblib
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import time
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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
# 配置类
# =====================
@dataclass
class Config:
    """配置类，集中管理所有参数"""
    # 技术指标参数
    RSI: int = 14
    EMA: int = 200
    ATR: int = 14
    MIN_BARS: int = 210
    FUTURE: int = 5
    
    # Walk-forward 参数
    TRAIN_WINDOW: int = 1000
    TEST_WINDOW: int = 200
    
    # 网格搜索配置
    USE_GRID_SEARCH: bool = True
    GRID_SEARCH_MODE: str = "first_only"  # "first_only", "all", "skip"
    CV_FOLDS: int = 3
    
    # 文件路径
    DATA_FILE: str = "XAUUSD.csv"
    MODEL_FILE: str = "xgb_walkforward_models.pkl"
    VISUALIZATION_FILE: str = "training_visualization.png"
    PARAMS_FILE: str = "best_params_history.csv"
    
    # 特征列表
    FEATURES: List[str] = None
    
    def __post_init__(self):
        if self.FEATURES is None:
            self.FEATURES = ["rsi", "atr", "ret_5", "vol", "ema_dist"]
    
    # 参数网格
    @property
    def param_grid(self) -> Dict:
        return {
            'n_estimators': [500, 800, 1000],
            'max_depth': [10, 15, 20],
            'learning_rate': [0.001, 0.003, 0.005],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
    
    # 默认参数
    @property
    def default_params(self) -> Dict:
        return {
            'n_estimators': 1000,
            'max_depth': 20,
            'learning_rate': 0.001,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'random_state': 42
        }

# =====================
# 数据加载和特征工程函数
# =====================
def load_data(file_path: str) -> pd.DataFrame:
    """加载CSV数据文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
    
    try:
        df = pd.read_csv(
            file_path,
            header=None,
            names=["date", "time", "open", "high", "low", "close", "volume"]
        )
        df.columns = [c.lower() for c in df.columns]
        
        # 数据验证
        required_columns = ["open", "high", "low", "close"]
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必需的列: {missing_cols}")
        
        if len(df) < 210:
            raise ValueError(f"数据量不足，至少需要210行，当前只有{len(df)}行")
        
        logger.info(f"成功加载数据: {len(df)} 行")
        return df
    except pd.errors.EmptyDataError:
        raise ValueError(f"数据文件为空: {file_path}")
    except Exception as e:
        raise RuntimeError(f"加载数据时出错: {str(e)}")


def create_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """创建技术指标特征"""
    cl = df["close"].values
    hi = df["high"].values
    lo = df["low"].values
    
    # 计算技术指标
    df["rsi"] = talib.RSI(cl, config.RSI)
    df["ema200"] = talib.EMA(cl, config.EMA)
    df["atr"] = talib.ATR(hi, lo, cl, config.ATR)
    df["ret_5"] = (df["close"] - df["close"].shift(5)) / df["close"].shift(5)
    df["vol"] = df["close"].rolling(10).std()
    df["ema_dist"] = (df["close"] - df["ema200"]) / df["ema200"]
    
    # 创建目标变量
    df["y"] = (df["close"].shift(-config.FUTURE) > df["close"]).astype(int)
    
    # 清理数据
    df = df.iloc[config.MIN_BARS:-config.FUTURE].dropna()
    
    # 验证特征列是否存在
    missing_features = [f for f in config.FEATURES if f not in df.columns]
    if missing_features:
        raise ValueError(f"缺少特征列: {missing_features}")
    
    logger.info(f"特征工程完成，剩余数据: {len(df)} 行")
    return df

# =====================
# 模型训练函数
# =====================
def perform_grid_search(train_data: pd.DataFrame, config: Config) -> Dict:
    """执行网格搜索找到最佳参数"""
    logger.info("开始网格搜索...")
    start_time = time.time()
    
    f1_scorer = make_scorer(f1_score, zero_division=0)
    base_model = XGBClassifier(
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )
    
    tscv = TimeSeriesSplit(n_splits=config.CV_FOLDS)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=config.param_grid,
        scoring=f1_scorer,
        cv=tscv,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(train_data[config.FEATURES], train_data["y"])
    
    best_params = {
        **grid_search.best_params_,
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'random_state': 42
    }
    
    elapsed_time = time.time() - start_time
    logger.info(f"网格搜索完成 (耗时: {elapsed_time:.2f}秒)")
    logger.info(f"最佳参数: {best_params}")
    logger.info(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")
    
    return best_params


def train_model(train_data: pd.DataFrame, params: Dict, config: Config) -> XGBClassifier:
    """训练单个模型"""
    model = XGBClassifier(**params)
    model.fit(train_data[config.FEATURES], train_data["y"])
    return model


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """计算评估指标"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }


def walk_forward_training(df: pd.DataFrame, config: Config) -> Tuple[List, Dict]:
    """执行Walk-Forward训练"""
    logger.info("=" * 60)
    logger.info("开始 Walk-Forward 训练...")
    if config.USE_GRID_SEARCH and config.GRID_SEARCH_MODE != "skip":
        logger.info(f"网格搜索模式: {config.GRID_SEARCH_MODE}")
        grid_size = np.prod([len(v) for v in config.param_grid.values()])
        logger.info(f"参数网格大小: {grid_size} 种组合")
    logger.info("=" * 60)
    
    # 初始化结果存储
    results = {
        'models': [],
        'best_params_history': [],
        'train_metrics': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
        'test_metrics': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
        'model_indices': []
    }
    
    best_params = config.default_params.copy()
    start = 0
    model_idx = 0
    
    while start + config.TRAIN_WINDOW + config.TEST_WINDOW < len(df):
        train = df.iloc[start:start+config.TRAIN_WINDOW]
        test = df.iloc[start+config.TRAIN_WINDOW:start+config.TRAIN_WINDOW+config.TEST_WINDOW]
        
        # 决定是否进行网格搜索
        should_grid_search = (
            config.USE_GRID_SEARCH and 
            config.GRID_SEARCH_MODE != "skip" and 
            (config.GRID_SEARCH_MODE == "all" or 
             (config.GRID_SEARCH_MODE == "first_only" and model_idx == 0))
        )
        
        if should_grid_search:
            logger.info(f"\n模型 {model_idx + 1}: 正在进行网格搜索...")
            best_params = perform_grid_search(train, config)
        else:
            if model_idx == 0 and not config.USE_GRID_SEARCH:
                logger.info(f"模型 {model_idx + 1}: 使用默认参数")
            else:
                logger.info(f"模型 {model_idx + 1}: 使用参数 "
                          f"n_estimators={best_params.get('n_estimators')}, "
                          f"max_depth={best_params.get('max_depth')}, "
                          f"learning_rate={best_params.get('learning_rate')}")
        
        # 训练模型
        model = train_model(train, best_params, config)
        results['models'].append(model)
        results['best_params_history'].append(best_params.copy())
        
        # 预测和评估
        train_pred = model.predict(train[config.FEATURES])
        test_pred = model.predict(test[config.FEATURES])
        
        train_metrics = calculate_metrics(train["y"].values, train_pred)
        test_metrics = calculate_metrics(test["y"].values, test_pred)
        
        # 保存指标
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            results['train_metrics'][metric].append(train_metrics[metric])
            results['test_metrics'][metric].append(test_metrics[metric])
        
        results['model_indices'].append(model_idx)
        
        # 打印进度
        logger.info(f"模型 {model_idx + 1}: 训练集准确率={train_metrics['accuracy']:.4f}, "
                   f"测试集准确率={test_metrics['accuracy']:.4f}, "
                   f"测试集F1={test_metrics['f1']:.4f}")
        
        start += config.TEST_WINDOW
        model_idx += 1
    
    logger.info(f"\n训练完成，共训练 {len(results['models'])} 个模型")
    return results['models'], results


def save_models(models: List[XGBClassifier], file_path: str) -> None:
    """保存模型到文件"""
    try:
        joblib.dump(models, file_path)
        logger.info(f"已保存 {len(models)} 个 walk-forward 模型到 {file_path}")
    except Exception as e:
        logger.error(f"保存模型失败: {str(e)}")
        raise

# =====================
# 可视化函数
# =====================
def plot_metric_comparison(ax, model_indices: List[int], train_values: List[float], 
                          test_values: List[float], metric_name: str, ylabel: str):
    """绘制训练集和测试集指标对比"""
    ax.plot(model_indices, train_values, 'o-', label=f'训练集{metric_name}', 
           linewidth=2, markersize=6)
    ax.plot(model_indices, test_values, 's-', label=f'测试集{metric_name}', 
           linewidth=2, markersize=6)
    ax.set_xlabel('模型编号', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'训练集 vs 测试集{metric_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])


def plot_feature_importance(ax, features: List[str], importance: np.ndarray, 
                           title: str, color: str = 'steelblue'):
    """绘制特征重要性"""
    feature_df = pd.DataFrame({
        'feature': features,
        'importance': importance
    }).sort_values('importance', ascending=True)
    ax.barh(feature_df['feature'], feature_df['importance'], color=color)
    ax.set_xlabel('重要性', fontsize=12)
    ax.set_ylabel('特征', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')


def create_training_visualization(results: Dict, config: Config, 
                                 file_path: Optional[str] = None) -> None:
    """创建训练结果可视化图表"""
    logger.info("生成可视化图表...")
    
    file_path = file_path or config.VISUALIZATION_FILE
    model_indices = results['model_indices']
    train_metrics = results['train_metrics']
    test_metrics = results['test_metrics']
    models = results['models']
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # 1-4. 指标对比图
    plot_metric_comparison(axes[0, 0], model_indices, train_metrics['accuracy'], 
                          test_metrics['accuracy'], '准确率', '准确率')
    plot_metric_comparison(axes[0, 1], model_indices, train_metrics['precision'], 
                          test_metrics['precision'], '精确率', '精确率')
    plot_metric_comparison(axes[1, 0], model_indices, train_metrics['recall'], 
                          test_metrics['recall'], '召回率', '召回率')
    plot_metric_comparison(axes[1, 1], model_indices, train_metrics['f1'], 
                          test_metrics['f1'], 'F1分数', 'F1分数')
    
    # 5. 最后一个模型的特征重要性
    plot_feature_importance(axes[2, 0], config.FEATURES, 
                           models[-1].feature_importances_,
                           '特征重要性（最后一个模型）', 'steelblue')
    
    # 6. 平均特征重要性
    avg_importance = np.mean([m.feature_importances_ for m in models], axis=0)
    plot_feature_importance(axes[2, 1], config.FEATURES, avg_importance,
                           '所有模型的平均特征重要性', 'coral')
    
    plt.tight_layout()
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    logger.info(f"可视化图表已保存为: {file_path}")
    plt.show()

def create_parameters_visualization(results: Dict, config: Config, 
                                   file_path: Optional[str] = None) -> None:
    """创建参数变化可视化图表"""
    if not (config.USE_GRID_SEARCH and config.GRID_SEARCH_MODE == "all" and 
            len(results['best_params_history']) > 0):
        return
    
    logger.info("生成参数变化可视化图表...")
    
    file_path = file_path or "parameters_visualization.png"
    best_params_history = results['best_params_history']
    model_indices = results['model_indices']
    test_f1s = results['test_metrics']['f1']
    
    # 提取参数值
    param_lists = {
        'n_estimators': [p.get('n_estimators', 0) for p in best_params_history],
        'max_depth': [p.get('max_depth', 0) for p in best_params_history],
        'learning_rate': [p.get('learning_rate', 0) for p in best_params_history],
        'subsample': [p.get('subsample', 0) for p in best_params_history],
        'colsample_bytree': [p.get('colsample_bytree', 0) for p in best_params_history]
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 参数配置
    param_configs = [
        ('n_estimators', '树的数量变化', 'steelblue', 'o-'),
        ('max_depth', '最大深度变化', 'coral', 's-'),
        ('learning_rate', '学习率变化', 'green', '^-'),
        ('subsample', '子样本比例变化', 'purple', 'd-'),
        ('colsample_bytree', '列采样比例变化', 'orange', 'v-')
    ]
    
    # 绘制前5个参数
    for idx, (param_name, title, color, style) in enumerate(param_configs):
        row, col = idx // 3, idx % 3
        axes[row, col].plot(model_indices, param_lists[param_name], style, 
                           color=color, linewidth=2, markersize=6)
        axes[row, col].set_xlabel('模型编号', fontsize=12)
        axes[row, col].set_ylabel(param_name, fontsize=12)
        axes[row, col].set_title(title, fontsize=14, fontweight='bold')
        axes[row, col].grid(True, alpha=0.3)
    
    # 参数与性能关系
    scatter = axes[1, 2].scatter(param_lists['n_estimators'], test_f1s, 
                                 c=param_lists['max_depth'], 
                                 s=100, alpha=0.6, cmap='viridis')
    axes[1, 2].set_xlabel('n_estimators', fontsize=12)
    axes[1, 2].set_ylabel('测试集F1分数', fontsize=12)
    axes[1, 2].set_title('参数与性能关系', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=axes[1, 2], label='max_depth')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    logger.info(f"参数可视化图表已保存为: {file_path}")
    plt.show()

# =====================
# 统计和报告函数
# =====================
def print_statistics(results: Dict, config: Config) -> None:
    """打印训练统计摘要"""
    logger.info("=" * 60)
    logger.info("训练统计摘要")
    logger.info("=" * 60)
    logger.info(f"总模型数: {len(results['models'])}")
    
    if config.USE_GRID_SEARCH and config.GRID_SEARCH_MODE != "skip":
        logger.info(f"网格搜索模式: {config.GRID_SEARCH_MODE}")
        if config.GRID_SEARCH_MODE == "first_only":
            logger.info(f"使用的参数: {results['best_params_history'][0]}")
    
    test_metrics = results['test_metrics']
    logger.info("\n测试集性能统计:")
    for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
        values = test_metrics[metric_name]
        mean_val = np.mean(values)
        std_val = np.std(values)
        metric_display = {'accuracy': '准确率', 'precision': '精确率', 
                         'recall': '召回率', 'f1': 'F1分数'}[metric_name]
        logger.info(f"  平均{metric_display}: {mean_val:.4f} ± {std_val:.4f}")
    
    # 最佳模型
    best_idx = np.argmax(test_metrics['f1'])
    logger.info(f"\n最佳模型 (按测试集F1):")
    logger.info(f"  模型编号: {best_idx}")
    logger.info(f"  测试集准确率: {test_metrics['accuracy'][best_idx]:.4f}")
    logger.info(f"  测试集F1: {test_metrics['f1'][best_idx]:.4f}")
    if config.USE_GRID_SEARCH and config.GRID_SEARCH_MODE == "all":
        logger.info(f"  使用的参数: {results['best_params_history'][best_idx]}")
    logger.info("=" * 60)


def save_params_history(results: Dict, config: Config, file_path: Optional[str] = None) -> None:
    """保存参数历史到CSV文件"""
    if not (config.USE_GRID_SEARCH and config.GRID_SEARCH_MODE != "skip"):
        return
    
    file_path = file_path or config.PARAMS_FILE
    try:
        params_df = pd.DataFrame(results['best_params_history'])
        params_df.to_csv(file_path, index=False)
        logger.info(f"参数历史已保存到: {file_path}")
    except Exception as e:
        logger.error(f"保存参数历史失败: {str(e)}")
        raise


# =====================
# 主函数
# =====================
def main():
    """主函数"""
    # 创建配置
    config = Config()
    
    try:
        # 加载数据
        df = load_data(config.DATA_FILE)
        
        # 特征工程
        df = create_features(df, config)
        
        # Walk-Forward训练
        models, results = walk_forward_training(df, config)
        results['models'] = models  # 确保models在results中
        
        # 保存模型
        save_models(models, config.MODEL_FILE)
        
        # 可视化
        create_training_visualization(results, config)
        create_parameters_visualization(results, config)
        
        # 打印统计摘要
        print_statistics(results, config)
        
        # 保存参数历史
        save_params_history(results, config)
        
        logger.info("所有任务完成！")
        
    except FileNotFoundError as e:
        logger.error(f"文件错误: {str(e)}")
        raise
    except ValueError as e:
        logger.error(f"数据错误: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"发生未知错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()
