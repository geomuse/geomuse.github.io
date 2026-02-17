"""
数据加载模块
支持从CSV文件加载历史数据和从MT5下载数据
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
import os


class DataLoader:
    """数据加载器"""
    
    def __init__(self):
        """初始化数据加载器"""
        pass
    
    def load_csv(self, filepath: str, 
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None) -> pd.DataFrame:
        """
        从CSV文件加载历史数据
        
        Args:
            filepath: CSV文件路径
            start_date: 起始日期 (格式: 'YYYY-MM-DD)
            end_date: 结束日期 (格式: 'YYYY-MM-DD')
            
        Returns:
            包含OHLC数据的DataFrame
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"数据文件不存在: {filepath}")
        
        # 先读取第一行检查是否有标题
        first_line = pd.read_csv(filepath, nrows=1, header=None)
        has_header = False
        
        # 检查第一行第一个值是否像日期
        try:
            first_val = str(first_line.iloc[0, 0])
            # 如果第一个值是数字日期（如20250110）或日期字符串，则没有标题
            if first_val.isdigit() and len(first_val) == 8:
                has_header = False
            elif 'date' in first_val.lower() or 'time' in first_val.lower():
                has_header = True
        except:
            pass
        
        # 根据是否有标题读取CSV
        if has_header:
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.lower()
        else:
            # 没有标题，手动指定列名
            df = pd.read_csv(filepath, header=None)
            # 根据列数推断格式
            if len(df.columns) == 7:
                df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
            elif len(df.columns) == 6:
                df.columns = ['date', 'time', 'open', 'high', 'low', 'close']
            else:
                # 默认假设前两列是日期和时间
                df.columns = ['date', 'time'] + [f'col_{i}' for i in range(len(df.columns)-2)]
        
        # 处理时间列
        df = self._process_datetime(df)
        
        # 确保必要的列存在
        required_cols = ['datetime', 'open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"CSV文件缺少必要的列: {missing_cols}")
        
        # 按时间排序
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # 过滤日期范围
        if start_date:
            df = df[df['datetime'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['datetime'] <= pd.to_datetime(end_date)]
        
        # 重置索引
        df = df.reset_index(drop=True)
        
        print(f"✓ 成功加载数据: {len(df)} 根K线")
        print(f"  时间范围: {df['datetime'].min()} 至 {df['datetime'].max()}")
        
        return df
    
    def _process_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理日期时间列,统一为datetime列
        
        Args:
            df: DataFrame
            
        Returns:
            处理后的DataFrame
        """
        # 情况1: 已有datetime列
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        # 情况2: 有date和time列
        elif 'date' in df.columns and 'time' in df.columns:
            # 组合date和time列，支持YYYYMMDD格式
            df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str),
                                           format='%Y%m%d %H:%M:%S', errors='coerce')
            # 如果上面失败，尝试其他格式
            if df['datetime'].isnull().any():
                df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
            df = df.drop(['date', 'time'], axis=1)
        
        # 情况3: 只有date列
        elif 'date' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'])
            df = df.drop(['date'], axis=1)
        
        else:
            raise ValueError("无法识别时间列，请确保CSV包含datetime、date或date+time列")
        
        return df
    
    def load_from_mt5(self, symbol: str, timeframe: str, 
                      start_date: str, end_date: str,
                      save_path: Optional[str] = None) -> pd.DataFrame:
        """
        从MT5下载历史数据
        
        Args:
            symbol: 交易品种
            timeframe: 时间周期 (如 '5min', '1H', '1D')
            start_date: 起始日期
            end_date: 结束日期
            save_path: 保存CSV文件路径（可选）
            
        Returns:
            包含OHLC数据的DataFrame
        """
        try:
            import MetaTrader5 as mt5
        except ImportError:
            raise ImportError("请安装MetaTrader5库: pip install MetaTrader5")
        
        # 初始化MT5
        if not mt5.initialize():
            raise Exception(f"MT5初始化失败: {mt5.last_error()}")
        
        try:
            # 转换时间周期
            tf_map = {
                '1min': mt5.TIMEFRAME_M1,
                '5min': mt5.TIMEFRAME_M5,
                '15min': mt5.TIMEFRAME_M15,
                '30min': mt5.TIMEFRAME_M30,
                '1H': mt5.TIMEFRAME_H1,
                '4H': mt5.TIMEFRAME_H4,
                '1D': mt5.TIMEFRAME_D1,
            }
            mt5_timeframe = tf_map.get(timeframe)
            if mt5_timeframe is None:
                raise ValueError(f"不支持的时间周期: {timeframe}")
            
            # 转换日期
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            
            # 下载数据
            rates = mt5.copy_rates_range(symbol, mt5_timeframe, start, end)
            
            if rates is None or len(rates) == 0:
                raise Exception(f"无法获取数据: {mt5.last_error()}")
            
            # 转换为DataFrame
            df = pd.DataFrame(rates)
            df['datetime'] = pd.to_datetime(df['time'], unit='s')
            df = df[['datetime', 'open', 'high', 'low', 'close', 'tick_volume']]
            df.rename(columns={'tick_volume': 'volume'}, inplace=True)
            
            print(f"✓ 从MT5下载数据成功: {len(df)} 根K线")
            
            # 保存到CSV
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                df.to_csv(save_path, index=False)
                print(f"✓ 数据已保存至: {save_path}")
            
            return df
            
        finally:
            mt5.shutdown()
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        验证数据质量
        
        Args:
            df: 数据DataFrame
            
        Returns:
            是否通过验证
        """
        issues = []
        
        # 检查缺失值
        if df.isnull().any().any():
            null_cols = df.columns[df.isnull().any()].tolist()
            issues.append(f"存在缺失值，列: {null_cols}")
        
        # 检查价格逻辑
        invalid_prices = (df['high'] < df['low']) | \
                        (df['high'] < df['open']) | \
                        (df['high'] < df['close']) | \
                        (df['low'] > df['open']) | \
                        (df['low'] > df['close'])
        
        if invalid_prices.any():
            issues.append(f"存在不合理的价格数据，{invalid_prices.sum()} 根K线")
        
        # 检查时间连续性
        if len(df) > 1:
            time_diffs = df['datetime'].diff().dropna()
            mode_diff = time_diffs.mode()[0] if len(time_diffs.mode()) > 0 else None
            if mode_diff:
                gaps = time_diffs[time_diffs > mode_diff * 2]
                if len(gaps) > 0:
                    issues.append(f"存在时间间隙: {len(gaps)} 处")
        
        if issues:
            print("⚠ 数据验证发现问题:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        
        print("✓ 数据验证通过")
        return True
