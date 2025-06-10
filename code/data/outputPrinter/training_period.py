import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import pandas as pd
import seaborn as sns
from pathlib import Path

class TrainingVisualizer:
    """训练过程可视化器
    
    用于记录和可视化训练过程中的各项指标:
    1. 奖励曲线
    2. 损失函数变化
    3. 策略性能指标
    4. 决策分布
    """
    
    def __init__(self, exp_dir: str):
        """初始化可视化器
        
        Args:
            exp_dir: 实验结果保存目录
        """
        self.exp_dir = Path(exp_dir)
        self.fig_dir = self.exp_dir / "figures"
        self.fig_dir.mkdir(exist_ok=True)
        
        # 设置绘图风格
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
    def plot_rewards(self, rewards: List[float], episode: int):
        """绘制奖励曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(rewards, label='Episode Reward')
        plt.plot(pd.Series(rewards).rolling(10).mean(), 
                label='Moving Average (10 episodes)')
        
        plt.title(f'Training Rewards (Episode {episode})')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(self.fig_dir / f'rewards_ep{episode}.png')
        plt.close()
        
    def plot_losses(self, training_stats: Dict, episode: int):
        """绘制损失函数曲线"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        
        # Meta Controller损失
        meta_losses = [x['total_loss'] for x in training_stats['meta_losses']]
        axes[0].plot(meta_losses, label='Loss')
        axes[0].plot(pd.Series(meta_losses).rolling(50).mean(), 
                    label='Moving Average')
        axes[0].set_title('Meta Controller Loss')
        axes[0].set_xlabel('Update Step')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Scheduling Policy损失
        sched_losses = [x['policy_loss'] for x in training_stats['scheduling_losses']]
        axes[1].plot(sched_losses, label='Loss')
        axes[1].plot(pd.Series(sched_losses).rolling(50).mean(), 
                    label='Moving Average')
        axes[1].set_title('Scheduling Policy Loss')
        axes[1].set_xlabel('Update Step')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        # Dispatching Policy损失
        disp_losses = [x['policy_loss'] for x in training_stats['dispatching_losses']]
        axes[2].plot(disp_losses, label='Loss')
        axes[2].plot(pd.Series(disp_losses).rolling(50).mean(), 
                    label='Moving Average')
        axes[2].set_title('Dispatching Policy Loss')
        axes[2].set_xlabel('Update Step')
        axes[2].set_ylabel('Loss')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / f'losses_ep{episode}.png')
        plt.close()
        
    def plot_decision_distribution(self, training_stats: Dict, episode: int):
        """绘制决策分布"""
        meta_actions = [x['action'].item() for x in training_stats['meta_decisions']]
        action_counts = pd.Series(meta_actions).value_counts()
        
        plt.figure(figsize=(8, 6))
        sns.barplot(x=action_counts.index, y=action_counts.values)
        plt.title(f'Meta Controller Decision Distribution (Episode {episode})')
        plt.xlabel('Action Type (0: Scheduling, 1: Dispatching)')
        plt.ylabel('Count')
        
        plt.savefig(self.fig_dir / f'decisions_ep{episode}.png')
        plt.close()
        
    def save_statistics(self, training_stats: Dict, episode: int):
        """保存训练统计数据"""
        stats_df = pd.DataFrame({
            'episode': range(episode + 1),
            'reward': training_stats['episode_rewards'],
            'meta_loss': [np.mean([x['total_loss'] for x in batch]) 
                         if batch else np.nan 
                         for batch in training_stats['meta_losses']],
            'scheduling_loss': [np.mean([x['policy_loss'] for x in batch]) 
                              if batch else np.nan 
                              for batch in training_stats['scheduling_losses']],
            'dispatching_loss': [np.mean([x['policy_loss'] for x in batch]) 
                               if batch else np.nan 
                               for batch in training_stats['dispatching_losses']]
        })
        
        stats_df.to_csv(self.exp_dir / 'training_stats.csv', index=False)