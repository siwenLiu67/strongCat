"""
混合规则DQN与派遣启发式算法的实验运行器
结合了规则选择的DQN和派遣启发式算法的混合方法
"""

import sys
import os
import time
from pathlib import Path
import numpy as np
import torch
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

# 添加项目根目录到路径
sys.path.append('./')
from model.environment import WarehouseEnvironment
from agents.rule_based_agent import RuleBasedDQNAgent
from agents.dispatch_heuristic import DispatchHeuristic
from agents.high_level_agent import HighLevelAgent as MetaAgent
from model.config import Config
from results.algorithm_results_saver import (
    save_algorithm_results_csv,
    set_random_seed, 
    generate_instance_id
)
from instances.case_generator import FlexibleJobShopScenario
from utils.metrics import ExperimentMetrics
from experiments.base_runner import BaseExperimentRunner

class HybridExperimentRunner(BaseExperimentRunner):
    """混合算法实验运行器"""
    
    def __init__(self, config: Config, env: WarehouseEnvironment):
        super().__init__(config, env)
        
        # 初始化智能体
        self.meta_agent = MetaAgent(config)
        self.rule_dqn_agent = RuleBasedDQNAgent(config)
        self.dispatch_agent = DispatchHeuristic()
        
        # 元策略训练缓冲
        self.meta_buffer = {
            'states': [], 
            'actions': [], 
            'rewards': [],
            'log_probs': [],
            'returns': []
        }
    def run_single_episode(self) -> ExperimentMetrics:
        """运行单个回合"""
        metrics = ExperimentMetrics()
        state = self.env.reset()
        done = False
        
        # 回合数据收集
        episode_data = defaultdict(list)
        start_time = time.time()
        
        try:
            while not done:
                # 元策略选择智能体
                agent_idx, log_prob, _ = self.meta_agent.select_action(state)
                
                # 根据选择的智能体执行动作
                action = self._execute_agent_action(agent_idx, state)
                
                # 环境步进
                next_state, reward, done, info = self.env.step(action)
                
                # 更新规则DQN（如果使用）
                if agent_idx == 0:
                    self._update_rule_dqn(state, action, reward, next_state, done)
                
                # 收集回合数据
                episode_data['states'].append(state)
                episode_data['actions'].append(agent_idx)
                episode_data['rewards'].append(reward)
                episode_data['log_probs'].append(log_prob)
                
                state = next_state
                metrics.episode_reward += reward
                metrics.episode_length += 1
            
            # 计算累积回报
            returns = self.calculate_returns(episode_data['rewards'])
            
            # 更新元策略
            metrics.meta_loss = self._update_meta_agent(episode_data, returns)
            
            # 更新其他指标
            metrics.running_time = time.time() - start_time
            metrics.makespan = getattr(self.env, 't', getattr(self.env, 'current_time', 0))
            metrics.tardy_penalty = getattr(self.env, 'tardy_penalty', 0)
            metrics.total_tardiness = getattr(self.env, 'total_weighted_tardiness', 0)
            metrics.machine_utilization = self.calculate_machine_utilization()
            
        except Exception as e:
            print(f"回合执行出错: {str(e)}")
        
        return metrics
    
    def _execute_agent_action(self, agent_idx: int, state: Dict) -> Dict:
        """执行选定智能体的动作"""
        if agent_idx == 0:
            action, _ = self.rule_dqn_agent.select_action(state)
        elif agent_idx == 1:
            action = self.dispatch_agent.select_action(state)
        else:
            action = {'wait': True}
        return action
    
    def _update_rule_dqn(self, state: Dict, action: Dict, 
                        reward: float, next_state: Dict, done: bool):
        """更新规则DQN智能体"""
        transition = {
            'states': [state],
            'actions': [action.get('rule_idx', 0)],
            'rewards': [reward],
            'next_states': [next_state],
            'dones': [done]
        }
        self.rule_dqn_agent.update(transition)
    
    def _update_meta_agent(self, episode_data: Dict, returns: torch.Tensor) -> float:
        """更新元策略智能体"""
        # 扩展回放缓冲区
        self.meta_buffer['states'].extend(episode_data['states'])
        self.meta_buffer['actions'].extend(episode_data['actions'])
        self.meta_buffer['log_probs'].extend(episode_data['log_probs'])
        self.meta_buffer['returns'].extend(returns.tolist())
        
        meta_loss = 0.0
        if len(self.meta_buffer['states']) >= self.batch_size:
            batch = {
                'states': self.meta_buffer['states'][:self.batch_size],
                'actions': self.meta_buffer['actions'][:self.batch_size],
                'log_probs': self.meta_buffer['log_probs'][:self.batch_size],
                'returns': torch.tensor(
                    self.meta_buffer['returns'][:self.batch_size], 
                    dtype=torch.float32
                )
            }
            meta_loss = self.meta_agent.update(batch)
            
            # 清除已使用的数据
            for key in self.meta_buffer:
                self.meta_buffer[key] = self.meta_buffer[key][self.batch_size:]
                
        return meta_loss
    


def run_experiment(config: Config, case: FlexibleJobShopScenario, 
                  seed: int, **kwargs) -> Dict:
    """运行实验的统一入口函数"""
    try:
        # 设置随机种子
        set_random_seed(seed)
        config.seeds = [seed]
        
        # 初始化环境和运行器
        env = WarehouseEnvironment(config, case)
        runner = HybridExperimentRunner(config, env)
        
        # 运行实验
        result = runner.run_experiment()
        stats = result['stats']
        total_time = result['total_time']
        
        # 计算额外指标
        additional_metrics = {
            "meta_agent_loss": float(np.mean(stats['meta_loss'][-10:])) if stats.get('meta_loss') else 0.0,
            "avg_tardy_penalty": float(np.mean(stats['tardy_penalty'])) if stats.get('tardy_penalty') else 0.0,
            "algorithm_type": "Hybrid_RuleDQN_DispatchHeuristic",
            "batch_size": runner.batch_size,
            "gamma": runner.gamma
        }
        
        # 保存结果
        instance_id = generate_instance_id(config)
        save_algorithm_results_csv(
            algo_name="RuleDQN+DispatchHeuristic",
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
