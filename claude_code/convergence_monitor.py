import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import torch
import os
from typing import Dict, List, Tuple, Optional

class ConvergenceMonitor:
    """收敛监控器：检测训练过程中的收敛状态"""
    
    def __init__(self, window_size: int = 50, patience: int = 20, min_improvement: float = 0.01):
        self.window_size = window_size
        self.patience = patience
        self.min_improvement = min_improvement
        
        # 收敛状态跟踪
        self.best_reward = -float('inf')
        self.no_improvement_count = 0
        self.converged = False
        
        # 滑动窗口统计
        self.reward_window = deque(maxlen=window_size)
        self.loss_window = deque(maxlen=window_size)
        self.success_window = deque(maxlen=window_size)
        
        # 历史记录
        self.history = {
            'episodes': [],
            'rewards': [],
            'losses': [],
            'success_rates': [],
            'strategic_goals': [],
            'policy_distribution': [],
            'learning_rates': []
        }
    
    def update(self, episode: int, stats: Dict) -> Dict:
        """更新监控器状态"""
        current_reward = stats.get('episode_reward', 0)
        current_loss = stats.get('meta_loss', 0)
        current_success = stats.get('strategic_success_rate', 0)
        
        # 更新滑动窗口
        self.reward_window.append(current_reward)
        self.loss_window.append(current_loss)
        self.success_window.append(current_success)
        
        # 更新历史记录
        self.history['episodes'].append(episode)
        self.history['rewards'].append(current_reward)
        self.history['losses'].append(current_loss)
        self.history['success_rates'].append(current_success)
        self.history['learning_rates'].append(stats.get('learning_rate', 0))
        
        # 检查收敛
        convergence_info = self._check_convergence(current_reward)
        
        return {
            'converged': self.converged,
            'best_reward': self.best_reward,
            'no_improvement_count': self.no_improvement_count,
            'window_stats': self._get_window_stats(),
            **convergence_info
        }
    
    def _check_convergence(self, current_reward: float) -> Dict:
        """检查收敛条件"""
        convergence_info = {
            'reward_improved': False,
            'reward_stable': False,
            'loss_stable': False,
            'suggestion': '继续训练'
        }
        
        # 检查奖励改进
        if current_reward > self.best_reward + self.min_improvement:
            self.best_reward = current_reward
            self.no_improvement_count = 0
            convergence_info['reward_improved'] = True
        else:
            self.no_improvement_count += 1
        
        # 检查奖励稳定性
        if len(self.reward_window) == self.window_size:
            reward_std = np.std(list(self.reward_window))
            reward_mean = np.mean(list(self.reward_window))
            convergence_info['reward_stable'] = reward_std < 0.1 * abs(reward_mean) if reward_mean != 0 else False
        
        # 检查损失稳定性
        if len(self.loss_window) == self.window_size:
            loss_std = np.std(list(self.loss_window))
            convergence_info['loss_stable'] = loss_std < 0.1
        
        # 收敛判断
        if self.no_improvement_count >= self.patience:
            self.converged = True
            convergence_info['suggestion'] = '考虑早停'
        
        return convergence_info
    
    def _get_window_stats(self) -> Dict:
        """获取滑动窗口统计"""
        if len(self.reward_window) == 0:
            return {}
        
        rewards = list(self.reward_window)
        losses = list(self.loss_window)
        successes = list(self.success_window)
        
        return {
            'reward_mean': np.mean(rewards),
            'reward_std': np.std(rewards),
            'reward_trend': self._calculate_trend(rewards),
            'loss_mean': np.mean(losses) if losses else 0,
            'loss_std': np.std(losses) if losses else 0,
            'success_mean': np.mean(successes) if successes else 0,
            'window_size': len(self.reward_window)
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """计算趋势（线性回归斜率）"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope
    
    def get_summary(self) -> Dict:
        """获取监控摘要"""
        return {
            'total_episodes': len(self.history['episodes']),
            'best_reward': self.best_reward,
            'converged': self.converged,
            'current_window_stats': self._get_window_stats(),
            'reward_range': (min(self.history['rewards']), max(self.history['rewards'])) if self.history['rewards'] else (0, 0)
        }


class TrainingVisualizer:
    """训练可视化生成器"""
    
    def __init__(self, output_dir: str = "results/convergence_plots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置绘图风格
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_convergence_dashboard(self, monitor: ConvergenceMonitor, instance_id: str) -> str:
        """创建收敛仪表板"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'收敛分析 - {instance_id}', fontsize=16, fontweight='bold')
        
        history = monitor.history
        episodes = history['episodes']
        
        # 1. 奖励曲线
        self._plot_reward_curve(axes[0, 0], episodes, history['rewards'], monitor.best_reward)
        
        # 2. 损失曲线
        if history['losses'] and any(l > 0 for l in history['losses']):
            self._plot_loss_curve(axes[0, 1], episodes, history['losses'])
        
        # 3. 战略成功率
        if history['success_rates']:
            self._plot_success_rate(axes[0, 2], episodes, history['success_rates'])
        
        # 4. 学习率变化
        if history['learning_rates']:
            self._plot_learning_rate(axes[1, 0], episodes, history['learning_rates'])
        
        # 5. 滑动窗口统计
        window_stats = self._calculate_window_stats(history['rewards'], monitor.window_size)
        self._plot_window_stats(axes[1, 1], window_stats)
        
        # 6. 收敛状态
        self._plot_convergence_status(axes[1, 2], monitor)
        
        plt.tight_layout()
        
        # 保存图片
        filename = os.path.join(self.output_dir, f"{instance_id}_convergence_dashboard.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def _plot_reward_curve(self, ax, episodes, rewards, best_reward):
        """绘制奖励曲线"""
        ax.plot(episodes, rewards, 'b-', alpha=0.7, linewidth=2, label='Episode Reward')
        ax.axhline(y=best_reward, color='r', linestyle='--', alpha=0.8, label=f'Best: {best_reward:.2f}')
        
        # 添加趋势线
        if len(rewards) > 10:
            z = np.polyfit(episodes, rewards, 1)
            p = np.poly1d(z)
            ax.plot(episodes, p(episodes), "r--", alpha=0.5, linewidth=1, label='Trend')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('奖励收敛曲线')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_loss_curve(self, ax, episodes, losses):
        """绘制损失曲线"""
        valid_losses = [l for l in losses if l > 0]
        valid_episodes = episodes[:len(valid_losses)]
        
        if valid_losses:
            ax.plot(valid_episodes, valid_losses, 'g-', alpha=0.7, linewidth=2)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Loss')
            ax.set_title('损失函数变化')
            ax.grid(True, alpha=0.3)
    
    def _plot_success_rate(self, ax, episodes, success_rates):
        """绘制战略成功率"""
        ax.plot(episodes, success_rates, 'purple', alpha=0.7, linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Success Rate')
        ax.set_title('战略目标成功率')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    def _plot_learning_rate(self, ax, episodes, learning_rates):
        """绘制学习率变化"""
        ax.semilogy(episodes, learning_rates, 'orange', alpha=0.7, linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Learning Rate (log)')
        ax.set_title('学习率调度')
        ax.grid(True, alpha=0.3)
    
    def _plot_window_stats(self, ax, window_stats):
        """绘制滑动窗口统计"""
        if not window_stats:
            ax.text(0.5, 0.5, '数据不足', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('滑动窗口统计')
            return
        
        episodes = list(range(len(window_stats['means'])))
        means = window_stats['means']
        stds = window_stats['stds']
        
        ax.plot(episodes, means, 'b-', alpha=0.7, linewidth=2, label='Mean')
        ax.fill_between(episodes, 
                       np.array(means) - np.array(stds),
                       np.array(means) + np.array(stds),
                       alpha=0.2, label='±1 Std')
        
        ax.set_xlabel('Window Index')
        ax.set_ylabel('Reward')
        ax.set_title('滑动窗口奖励统计')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_convergence_status(self, ax, monitor):
        """绘制收敛状态"""
        summary = monitor.get_summary()
        
        status_text = [
            f"总回合数: {summary['total_episodes']}",
            f"最佳奖励: {summary['best_reward']:.2f}",
            f"收敛状态: {'是' if summary['converged'] else '否'}",
            f"无改进计数: {monitor.no_improvement_count}/{monitor.patience}"
        ]
        
        if summary['current_window_stats']:
            stats = summary['current_window_stats']
            status_text.extend([
                f"窗口奖励均值: {stats['reward_mean']:.2f}",
                f"窗口奖励标准差: {stats['reward_std']:.2f}",
                f"奖励趋势: {stats['reward_trend']:.4f}"
            ])
        
        ax.text(0.1, 0.9, '\n'.join(status_text), transform=ax.transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('收敛状态摘要')
    
    def _calculate_window_stats(self, rewards, window_size):
        """计算滑动窗口统计"""
        if len(rewards) < window_size:
            return {}
        
        means = []
        stds = []
        
        for i in range(len(rewards) - window_size + 1):
            window = rewards[i:i + window_size]
            means.append(np.mean(window))
            stds.append(np.std(window))
        
        return {'means': means, 'stds': stds}


def create_simple_convergence_plot(rewards: List[float], output_path: str):
    """创建简单的收敛图"""
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, 'b-', alpha=0.7, linewidth=2, label='Episode Reward')
    
    # 添加滑动平均
    if len(rewards) > 10:
        window_size = min(20, len(rewards) // 5)
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards)), moving_avg, 'r-', 
                alpha=0.8, linewidth=2, label=f'{window_size}-Episode Moving Avg')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('训练收敛曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
