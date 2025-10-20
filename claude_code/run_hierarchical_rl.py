import sys
import os
import time
import pickle
import numpy as np
from collections import defaultdict
from typing import Dict
import torch

from environment import WarehouseEnvironment
from high_level_controller import HighLevelController
from low_level_policy import LowLevelPolicy
from data_structures import Job
from case_generator import FlexibleJobShopScenario
from config import Config
import torch
import torch.optim as optim
from algorithm_results_saver import save_algorithm_results_csv, generate_instance_id, set_random_seed

class HierarchicalRLAgent:
    """层次强化学习智能体 - 整合高层控制器和底层策略"""
    
    def __init__(self, config):
        self.config = config
        
        # 初始化高层控制器
        self.high_level_controller = HighLevelController(config)
        
        # 初始化底层策略
        self.scheduling_policy = LowLevelPolicy(config, policy_type="scheduling")
        self.dispatching_policy = LowLevelPolicy(config, policy_type="dispatching")
        
        # 训练参数
        self.high_level_update_freq = 10  # 高层控制器更新频率
        self.low_level_update_freq = 5    # 底层策略更新频率
        self.step_count = 0
        
        # 子目标跟踪
        self.current_subgoal = None
        self.subgoal_start_time = 0
        
    def select_action(self, state: Dict) -> Dict:
        """选择动作 - 层次决策过程"""
        self.step_count += 1
        
        # 检查是否需要高层决策（每10步或子目标完成时）
        if (self.step_count % self.high_level_update_freq == 0 or 
            self.current_subgoal is None):
            
            # 高层控制器选择动作
            high_level_action = self.high_level_controller.select_action(state)
            
            # 设定新的子目标
            self.current_subgoal = self.high_level_controller.set_subgoal(state, high_level_action)
            self.subgoal_start_time = state.get('current_time', 0)
            
            print(f"高层决策: 动作={high_level_action}, 子目标={self.current_subgoal}")
        
        # 底层策略执行动作
        if self.current_subgoal['type'] == 'scheduling':
            # 调度子目标 - 使用调度策略
            _, action = self.scheduling_policy.select_action(state, self.current_subgoal)
        elif self.current_subgoal['type'] == 'dispatching':
            # 配送子目标 - 使用配送策略
            _, action = self.dispatching_policy.select_action(state, self.current_subgoal)
        else:
            # 等待子目标 - 直接返回等待动作
            action = {'wait': True}
        
        return action
    
    def update(self, state: Dict, action: Dict, reward: float, next_state: Dict, done: bool, 
               subgoal_completed: bool, intrinsic_reward: float):
        """更新所有策略"""
        
        # 计算高层奖励（环境奖励 + 内在奖励）
        high_level_reward = reward + intrinsic_reward
        
        # 存储高层经验
        high_level_action = 0  # 简化处理，实际应该记录高层动作
        if self.current_subgoal:
            if self.current_subgoal['type'] == 'scheduling':
                high_level_action = 0
            elif self.current_subgoal['type'] == 'dispatching':
                high_level_action = 1
            else:
                high_level_action = 2
        
        self.high_level_controller.store_experience(
            state, high_level_action, high_level_reward, next_state, done
        )
        
        # 存储底层经验
        if 'schedule' in action:
            # 调度动作 - 存储到调度策略
            scheduling_state = state
            scheduling_reward = reward
            if subgoal_completed:
                scheduling_reward += intrinsic_reward
            
            # 这里需要记录调度动作的索引，简化处理
            scheduling_action_idx = 0
            self.scheduling_policy.store_experience(
                scheduling_state, scheduling_action_idx, scheduling_reward, next_state, done
            )
            
        elif 'dispatch' in action:
            # 配送动作 - 存储到配送策略
            dispatching_state = state
            dispatching_reward = reward
            if subgoal_completed:
                dispatching_reward += intrinsic_reward
            
            # 这里需要记录配送动作的索引，简化处理
            dispatching_action_idx = 0
            self.dispatching_policy.store_experience(
                dispatching_state, dispatching_action_idx, dispatching_reward, next_state, done
            )
        
        # 定期更新网络
        if self.step_count % self.high_level_update_freq == 0:
            high_level_loss = self.high_level_controller.update()
            if high_level_loss > 0:
                print(f"高层控制器损失: {high_level_loss:.4f}")
        
        if self.step_count % self.low_level_update_freq == 0:
            scheduling_loss = self.scheduling_policy.update()
            if scheduling_loss > 0:
                print(f"调度策略损失: {scheduling_loss:.4f}")
            
            dispatching_loss = self.dispatching_policy.update()
            if dispatching_loss > 0:
                print(f"配送策略损失: {dispatching_loss:.4f}")

def run_hierarchical_rl_experiment(config, case, seed, **kwargs):
    """
    层次强化学习实验运行函数
    """
    set_random_seed(seed)
    config.seed = seed
    instance_id = generate_instance_id(config)
    env = WarehouseEnvironment(config, case)
    agent = HierarchicalRLAgent(config)
    
    episodes = config.episodes
    stats = defaultdict(list)
    start_time = time.time()
    
    # 学习率跟踪
    learning_rates = []
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        # 重置子目标
        agent.current_subgoal = None
        
        while not done:
            # 选择动作
            action = agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 检查子目标完成情况
            subgoal_completed, intrinsic_reward = agent.high_level_controller.check_subgoal_completion(next_state)
            
            # 更新智能体
            agent.update(state, action, reward, next_state, done, subgoal_completed, intrinsic_reward)
            
            # 更新状态
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            # 子目标完成时打印信息
            if subgoal_completed:
                print(f"子目标完成! 内在奖励: {intrinsic_reward:.2f}")
        
        # 收集统计信息
        stats['episode_rewards'].append(episode_reward)
        stats['episode_lengths'].append(episode_length)
        stats['makespans'].append(getattr(env, 't', getattr(env, 'current_time', 0)))
        stats['running_times'].append(time.time() - start_time)
        stats['tardy_penalty'].append(getattr(env, 'tardy_penalty', 0))
        stats['total_tardiness'].append(getattr(env, 'total_weighted_tardiness', 0))
        stats['objective_value'].append(getattr(env, 'tardy_penalty', 0) + getattr(env, 'total_weighted_tardiness', 0))
        stats['machine_utilization'].append(calculate_machine_utilization(env))
        
        # 每10个episode输出信息
        if episode % 10 == 0:
            print(f'Episode {episode}, Reward: {episode_reward:.2f}, Steps: {episode_length}')
            print(f"  机器利用率: {stats['machine_utilization'][-1]:.2%}")
            print(f"  总延误: {stats['total_tardiness'][-1]:.2f}")
        
        # 调试信息（可选）
        if episode == 0 or episode == episodes - 1:
            job = env.dispatched_jobs[0] if env.dispatched_jobs else None
            distributor = env.distributors[job.distributor_id] if job and hasattr(job, 'distributor_id') else None
            print(f"Episode {episode} - Job: {job}, Distributor: {distributor}")

    total_time = time.time() - start_time
    
    additional_metrics = {
        "algorithm_type": "Hierarchical_RL",
        "high_level_update_freq": agent.high_level_update_freq,
        "low_level_update_freq": agent.low_level_update_freq,
        "avg_tardy_penalty": float(np.mean(stats['tardy_penalty'])),
        "final_machine_utilization": float(stats['machine_utilization'][-1])
    }
    
    save_algorithm_results_csv(
        algo_name="HierarchicalRL",
        instance_id=instance_id,
        seed=seed,
        config=config,
        stats=stats,
        env=env,
        total_time=total_time,
        additional_metrics=additional_metrics
    )
    
    result = {
        'stats': stats,
        'env': env,
        'additional_metrics': additional_metrics
    }
    return result

def calculate_machine_utilization(env) -> float:
    """计算机器利用率"""
    makespan = env.t if env.t > 0 else 1
    total_work_time = sum(getattr(m, 'total_busy_time', 0) for m in env.machines)
    total_available_time = makespan * len(env.machines)
    if total_available_time == 0:
        return 0.0
    return total_work_time / total_available_time

# 保留原main函数用于单独运行
def main():
    seed = 42
    set_random_seed(seed)
    config = Config()
    case = FlexibleJobShopScenario(config=config)
    result = run_hierarchical_rl_experiment(config, case, seed)
    print("层次强化学习实验完成，结果已保存。")

if __name__ == "__main__":
    main()
