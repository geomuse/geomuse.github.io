"""
性能指标计算模块
计算回测结果的各项性能指标并生成可视化图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import os


class PerformanceMetrics:
    """性能指标计算器"""
    
    def __init__(self, results: Dict):
        """
        初始化
        
        Args:
            results: 回测结果字典
        """
        self.results = results
        self.trades_df = pd.DataFrame(results['trades']) if results['trades'] else pd.DataFrame()
        self.equity_df = pd.DataFrame(results['equity_curve']) if results['equity_curve'] else pd.DataFrame()
    
    def calculate_all_metrics(self) -> Dict:
        """
        计算所有性能指标
        
        Returns:
            性能指标字典
        """
        metrics = {
            'basic': self._calculate_basic_metrics(),
            'risk': self._calculate_risk_metrics(),
            'trade_analysis': self._calculate_trade_analysis()
        }
        
        return metrics
    
    def _calculate_basic_metrics(self) -> Dict:
        """计算基本指标"""
        initial = self.results['initial_balance']
        final = self.results['final_balance']
        
        metrics = {
            'Initial Balance': f"${initial:,.2f}",
            'Final Balance': f"${final:,.2f}",
            'Net Profit': f"${final - initial:,.2f}",
            'Total Return': f"{self.results['total_return']:.2f}%",
            'Total Trades': self.results['total_trades'],
            'Winning Trades': self.results['winning_trades'],
            'Losing Trades': self.results['losing_trades'],
            'Win Rate': f"{self.results['win_rate']:.2f}%"
        }
        
        return metrics
    
    def _calculate_risk_metrics(self) -> Dict:
        """计算风险指标"""
        if self.equity_df.empty:
            return {}
        
        # 计算回撤
        equity = self.equity_df['equity'].values
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100
        max_drawdown = np.min(drawdown)
        
        # 计算收益率序列
        returns = self.equity_df['equity'].pct_change().dropna()
        
        # 夏普比率（假设无风险利率为0，年化）
        if len(returns) > 0 and returns.std() != 0:
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()  # 假设每年252个交易日
        else:
            sharpe_ratio = 0
        
        # 索提诺比率（只考虑下行波动）
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() != 0:
            sortino_ratio = np.sqrt(252) * returns.mean() / downside_returns.std()
        else:
            sortino_ratio = 0
        
        metrics = {
            'Max Drawdown': f"{max_drawdown:.2f}%",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}",
            'Sortino Ratio': f"{sortino_ratio:.2f}"
        }
        
        return metrics
    
    def _calculate_trade_analysis(self) -> Dict:
        """计算交易分析指标"""
        if self.trades_df.empty:
            return {}
        
        # 盈利交易和亏损交易
        winning_trades = self.trades_df[self.trades_df['profit'] > 0]
        losing_trades = self.trades_df[self.trades_df['profit'] < 0]
        
        # 平均盈利和平均亏损
        avg_win = winning_trades['profit'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['profit'].mean() if len(losing_trades) > 0 else 0
        
        # 盈亏比
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # 最大盈利和最大亏损
        max_win = self.trades_df['profit'].max() if len(self.trades_df) > 0 else 0
        max_loss = self.trades_df['profit'].min() if len(self.trades_df) > 0 else 0
        
        # 平均持仓时间
        if 'entry_time' in self.trades_df.columns and 'exit_time' in self.trades_df.columns:
            self.trades_df['holding_time'] = (
                pd.to_datetime(self.trades_df['exit_time']) - 
                pd.to_datetime(self.trades_df['entry_time'])
            )
            avg_holding_time = self.trades_df['holding_time'].mean()
        else:
            avg_holding_time = pd.Timedelta(0)
        
        metrics = {
            'Average Win': f"${avg_win:.2f}",
            'Average Loss': f"${avg_loss:.2f}",
            'Profit Factor': f"{profit_factor:.2f}",
            'Largest Win': f"${max_win:.2f}",
            'Largest Loss': f"${max_loss:.2f}",
            'Avg Holding Time': str(avg_holding_time)
        }
        
        return metrics
    
    def generate_report(self, save_path: str = None) -> str:
        """
        生成文本报告
        
        Args:
            save_path: 保存路径（可选）
            
        Returns:
            报告文本
        """
        metrics = self.calculate_all_metrics()
        
        report = []
        report.append("=" * 60)
        report.append("回测性能报告")
        report.append("=" * 60)
        
        # 基本指标
        report.append("\n基本指标:")
        report.append("-" * 60)
        for key, value in metrics['basic'].items():
            report.append(f"{key:.<40} {value:>15}")
        
        # 风险指标
        if metrics['risk']:
            report.append("\n风险指标:")
            report.append("-" * 60)
            for key, value in metrics['risk'].items():
                report.append(f"{key:.<40} {value:>15}")
        
        # 交易分析
        if metrics['trade_analysis']:
            report.append("\n交易分析:")
            report.append("-" * 60)
            for key, value in metrics['trade_analysis'].items():
                report.append(f"{key:.<40} {value:>15}")
        
        report.append("\n" + "=" * 60)
        
        report_text = "\n".join(report)
        
        # 保存报告
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"✓ 报告已保存至: {save_path}")
        
        return report_text
    
    def plot_results(self, save_path: str = None, show: bool = True):
        """
        绘制回测结果图表
        
        Args:
            save_path: 保存路径（可选）
            show: 是否显示图表
        """
        if self.equity_df.empty:
            print("无数据可绘制")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        fig.suptitle('EMA策略回测结果', fontsize=16, fontweight='bold')
        
        # 1. 权益曲线
        ax1 = axes[0]
        ax1.plot(self.equity_df['datetime'], self.equity_df['equity'], 
                label='Equity', linewidth=2, color='#2E86AB')
        ax1.plot(self.equity_df['datetime'], self.equity_df['balance'], 
                label='Balance', linewidth=1.5, color='#A23B72', alpha=0.7)
        ax1.axhline(y=self.results['initial_balance'], color='gray', 
                   linestyle='--', alpha=0.5, label='Initial Balance')
        ax1.set_ylabel('balance ($)', fontsize=12)
        ax1.set_title('equity', fontsize=14)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 2. 回撤曲线
        ax2 = axes[1]
        equity = self.equity_df['equity'].values
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100
        ax2.fill_between(self.equity_df['datetime'], drawdown, 0, 
                         color='#F18F01', alpha=0.6)
        ax2.set_ylabel('drawdown (%)', fontsize=12)
        ax2.set_title('drawdown curve', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # 3. 累计交易盈亏
        ax3 = axes[2]
        if not self.trades_df.empty:
            cumulative_profit = self.trades_df['profit'].cumsum()
            colors = ['#06A77D' if p > 0 else '#D62246' for p in self.trades_df['profit']]
            ax3.bar(range(len(self.trades_df)), self.trades_df['profit'], 
                   color=colors, alpha=0.7)
            ax3.plot(range(len(self.trades_df)), cumulative_profit, 
                    color='#2E86AB', linewidth=2, label='Cumulative P&L')
            ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax3.set_xlabel('trade number', fontsize=12)
            ax3.set_ylabel('profit ($)', fontsize=12)
            ax3.set_title('profit distribution', fontsize=14)
            ax3.legend(loc='best')
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 图表已保存至: {save_path}")
        
        # 显示图表
        if show:
            plt.show()
        else:
            plt.close()
