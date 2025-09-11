"""
实验运行器基类
提供通用的实验运行和数据收集功能
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import torch
import numpy as np
from collections import defaultdict
import sys
sys.path.append('./')
from model.config import Config
from model.environment import WarehouseEnvironment
from utils.metrics import ExperimentMetrics

class BaseExperimentRunner(ABC):
    """实验运行器基类"""
    
    def __init__(self, config: Config, env: WarehouseEnvironment):
        self.config = config
        self.env = env
        self.batch_size = getattr(config, "batch_size", 32)
        self.gamma = getattr(config, "gamma", 0.95)
        
    @abstractmethod
    def run_single_episode(self) -> ExperimentMetrics:
        """运行单个回合，需要由子类实现"""
        pass
    
    def calculate_returns(self, rewards: List[float]) -> torch.Tensor:
        """计算累积回报"""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns)
    
    def calculate_machine_utilization(self) -> float:
        """计算机器利用率"""
        makespan = self.env.t if self.env.t > 0 else 1
        total_work_time = sum(
            getattr(m, 'total_busy_time', 0) 
            for m in self.env.machines
        )
        total_available_time = makespan * len(self.env.machines)
        return total_work_time / total_available_time if total_available_time > 0 else 0.0
    
    def collect_episode_stats(self, metrics: ExperimentMetrics) -> Dict[str, list]:
        """收集回合统计数据"""
        stats = defaultdict(list)
        stats_dict = metrics.to_dict()
        for key, value in stats_dict.items():
            stats[key].append(value)
        return stats
    
    def print_progress(self, episode: int, total_episodes: int, metrics: ExperimentMetrics):
        """打印实验进度"""
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{total_episodes} - "
                  f"Reward: {metrics.episode_reward:.2f}, "
                  f"Makespan: {metrics.makespan:.2f}")
    
    def run_experiment(self, episodes: Optional[int] = None) -> Dict[str, Any]:
        """运行完整实验"""
        num_episodes = episodes if episodes is not None else getattr(self.config, "episodes", 1)
            
        stats = defaultdict(list)
        start_time = time.time()
        
        try:
            for episode in range(num_episodes):
                # 运行单个回合
                metrics = self.run_single_episode()
                
                # 收集统计数据
                episode_stats = self.collect_episode_stats(metrics)
                for key, values in episode_stats.items():
                    stats[key].extend(values)
                
                # 打印进度
                self.print_progress(episode, num_episodes, metrics)
            
            # 计算实验总时间
            total_time = time.time() - start_time
            
            return {
                'stats': stats,
                'total_time': total_time,
                'env': self.env
            }
            
        except Exception as e:
            print(f"实验执行出错: {str(e)}")
            return {
                'stats': {'error': str(e)},
                'total_time': time.time() - start_time,
                'env': None
            }
