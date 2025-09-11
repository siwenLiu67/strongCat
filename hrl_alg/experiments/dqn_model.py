"""
深度Q网络(DQN)实验运行器
用于解决集成柔性作业车间调度与派遣问题(IFJSSP-DP)
"""

import sys
import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Any, Optional

# 添加项目根目录到路径
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))

from model.environment import WarehouseEnvironment
from agents.dqn_agent import DQNAgent
from model.config import Config
from results.algorithm_results_saver import (
    save_algorithm_results_csv,
    set_random_seed, 
    generate_instance_id
)
from instances.case_generator import FlexibleJobShopScenario
from utils.metrics import ExperimentMetrics
from experiments.base_runner import BaseExperimentRunner

class DQNExperimentRunner(BaseExperimentRunner):
    """DQN算法实验运行器"""
    
    def __init__(self, config: Config, env: WarehouseEnvironment):
        super().__init__(config, env)
        
        # 初始化DQN智能体
        self.state_dim = 20  # 状态维度
        self.action_dim = 20000  # 动作空间大小
        self.dqn_agent = DQNAgent(self.state_dim, self.action_dim, config)
        
    def run_single_episode(self) -> ExperimentMetrics:
        """运行单个回合"""
        metrics = ExperimentMetrics()
        state = self.env.reset()
        done = False
        
        start_time = time.time()
        
        try:
            while not done:
                # 获取有效动作和状态向量
                available_actions = self.env.available_actions()
                valid_actions = available_actions if available_actions else [{'wait': True}]
                state_vector = self.dqn_agent.state_to_vector(state)
                
                # 选择动作索引
                action_indices = list(range(len(valid_actions)))
                action_idx = self.dqn_agent.select_action(state_vector, action_indices)
                
                # 执行动作
                action = valid_actions[action_idx]
                next_state, reward, done, info = self.env.step(action)
                
                # 存储经验并更新网络
                next_state_vector = self.dqn_agent.state_to_vector(next_state)
                self.dqn_agent.store_transition(state_vector, action_idx, reward, next_state_vector, done)
                loss = self.dqn_agent.update()
                
                # 更新状态和指标
                state = next_state
                metrics.episode_reward += reward
                metrics.episode_length += 1
                metrics.running_time = time.time() - start_time
                metrics.makespan = getattr(self.env, 't', getattr(self.env, 'current_time', 0))
                metrics.tardy_penalty = getattr(self.env, 'tardy_penalty', 0)
                metrics.total_tardiness = getattr(self.env, 'total_weighted_tardiness', 0)
                metrics.machine_utilization = self.calculate_machine_utilization()
            
                if loss is not None:
                    metrics.dqn_loss = loss
            
            # 更新其他指标
            metrics.running_time = time.time() - start_time
            metrics.makespan = getattr(self.env, 't', getattr(self.env, 'current_time', 0))
            metrics.tardy_penalty = getattr(self.env, 'tardy_penalty', 0)
            metrics.total_tardiness = getattr(self.env, 'total_weighted_tardiness', 0)
            metrics.machine_utilization = self.calculate_machine_utilization()
            
        except Exception as e:
            print(f"回合执行出错: {str(e)}")
            import traceback
            print(f"错误详情: {traceback.format_exc()}")
        
        return metrics

def run_experiment(config: Config, case: FlexibleJobShopScenario, 
                  seed: int, **kwargs) -> Dict:
    """运行实验的统一入口函数"""
    try:
        # 设置随机种子
        set_random_seed(seed)
        config.seeds = [seed]
        
        # 初始化环境和运行器
        env = WarehouseEnvironment(config, case)
        runner = DQNExperimentRunner(config, env)
        
        # 运行实验
        result = runner.run_experiment()
        stats = result['stats']
        total_time = result['total_time']
        
        # 添加必要的指标
        if 'objective_value' not in stats:
            stats['objective_value'] = []
        if 'total_weighted_tardiness' not in stats:
            stats['total_weighted_tardiness'] = []
            
        # 计算额外指标
        additional_metrics = {
            "dqn_loss": float(np.mean(stats['dqn_loss'][-10:])) if stats.get('dqn_loss') else 0.0,
            "avg_tardy_penalty": float(np.mean(stats['tardy_penalty'])) if stats.get('tardy_penalty') else 0.0,
            "algorithm_type": "DQN",
            "batch_size": runner.dqn_agent.batch_size,
            "gamma": runner.dqn_agent.gamma,
            "epsilon": runner.dqn_agent.epsilon,
            "total_tardiness": float(np.mean(stats['total_weighted_tardiness'])) if stats.get('total_weighted_tardiness') else 0.0,
            "objective_value": float(np.mean(stats['objective_value'])) if stats.get('objective_value') else 0.0
        }
        
        # 保存结果
        instance_id = generate_instance_id(config)
        save_algorithm_results_csv(
            algo_name="DQN",
            instance_id=instance_id,
            seed=seed,
            config=config,
            stats=stats,
            env=env,
            total_time=total_time,
            additional_metrics=additional_metrics
        )
        
        return {
            'stats': stats,
            'env': env,
            'additional_metrics': additional_metrics
        }
        
    except Exception as e:
        print(f"实验执行出错: {str(e)}")
        return {
            'stats': {'error': str(e)},
            'env': None,
            'additional_metrics': {'error': str(e)}
        }

def main():
    """主函数，用于单独运行测试"""
    try:
        # 配置实验参数
        seed = 42
        set_random_seed(seed)
        config = Config()
        
        # 生成测试场景
        print("生成测试场景...")
        case = FlexibleJobShopScenario(config=config)
        
        # 运行实验
        print("开始运行实验...")
        result = run_experiment(config, case, seed)
        
        print("\n实验完成!")
        if result['env'] is not None:
            print(f"最终目标值: {result['stats']['objective_value'][-1]:.2f}")
            print(f"平均奖励: {np.mean(result['stats']['episode_rewards']):.2f}")
            print(f"平均makespan: {np.mean(result['stats']['makespans']):.2f}")
        else:
            print(f"实验失败: {result['stats']['error']}")
            
    except Exception as e:
        print(f"程序执行出错: {str(e)}")

if __name__ == "__main__":
    main()