"""
PPO算法模块
基于近端策略优化的强化学习算法，遵循统一算法接口
"""

import numpy as np
import torch
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

from ..agents.ppo_agent import PPOAgent
from ..models.environment import WarehouseEnvironment
from ..data_loader import DataLoader
from ..config import Config


class PPOAlgorithm:
    """
    PPO算法类
    实现统一的算法接口，用于集成调度与配送问题
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.name = "PPO"
        self.config = config or Config()
        self.agent = None
        self.environment = None
        self.data_loader = None
        
    def solve(self, instance_data: Dict, algorithm_params: Dict = None) -> Dict:
        """
        使用PPO算法求解集成调度与配送问题
        
        Args:
            instance_data: 实例数据，包含作业、机器、配送商等信息
            algorithm_params: 算法参数
            
        Returns:
            包含调度结果和性能指标的结果字典
        """
        # 设置随机种子
        seed = algorithm_params.get('seed', 42) if algorithm_params else 42
        self._set_random_seed(seed)
        
        # 创建数据加载器和环境
        self.data_loader = DataLoader(self.config, instance_data)
        self.environment = WarehouseEnvironment(self.config, self.data_loader)
        
        # 初始化智能体
        state_dim = 20  # 状态维度
        action_dim = 20000  # 动作空间大小
        self.agent = PPOAgent(state_dim, action_dim, self.config)
        
        # 训练参数
        episodes = algorithm_params.get('episodes', 200) if algorithm_params else 200
        stats = defaultdict(list)
        
        print(f"开始PPO训练 {episodes} episodes...")
        
        # 训练循环
        for episode in range(episodes):
            state_dict = self.environment.reset()
            state = self._state_dict_to_array(state_dict)
            episode_reward = 0
            episode_length = 0
            
            while True:
                # 获取有效动作
                valid_actions = self._get_valid_actions()
                if not valid_actions:
                    valid_actions = [19999]  # 至少包含等待动作
                
                # 选择动作
                action, log_prob, value = self.agent.select_action(state, valid_actions)
                
                # 将动作转换为环境期望的格式
                action_dict = self._action_to_dict(action)
                
                # 执行动作
                next_state_dict, reward, done, info = self.environment.step(action_dict)
                next_state = self._state_dict_to_array(next_state_dict)
                
                # 存储经验
                self.agent.store_transition(state, action, log_prob, value, reward, done)
                
                # 更新状态和统计
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            # 更新网络
            loss = self.agent.update()
            
            # 记录统计数据
            stats['episode_rewards'].append(episode_reward)
            stats['episode_lengths'].append(episode_length)
            stats['makespans'].append(self.environment.t)
            stats['completed_jobs'].append(len(self.environment.completed_jobs))
            stats['dispatched_jobs'].append(len(self.environment.dispatched_jobs))
            
            # 打印进度
            if (episode + 1) % 20 == 0:
                avg_reward = np.mean(stats['episode_rewards'][-20:])
                avg_makespan = np.mean(stats['makespans'][-20:])
                print(f"Episode {episode + 1}/{episodes}")
                print(f"  平均奖励: {avg_reward:.2f}")
                print(f"  平均makespan: {avg_makespan:.2f}")
                print(f"  完成作业: {len(self.environment.completed_jobs)}/{len(self.environment.available_jobs) + len(self.environment.completed_jobs)}")
                print(f"  派遣作业: {len(self.environment.dispatched_jobs)}/{len(self.environment.available_jobs) + len(self.environment.completed_jobs)}")
        
        # 收集结果
        result = self._collect_results(stats, instance_data, episodes)
        return result
    
    def _set_random_seed(self, seed: int):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    def _state_dict_to_array(self, state_dict: Dict) -> np.ndarray:
        """
        将环境状态字典转换为numpy数组
        """
        state_vector = []
        
        # 添加当前时间
        state_vector.append(state_dict.get('current_time', 0.0))
        
        # 添加机器状态
        for machine in state_dict.get('machines', []):
            state_vector.append(machine.get('available', 1.0))
            state_vector.append(machine.get('current_job_id', -1))
            state_vector.append(machine.get('remaining_time', 0.0))
        
        # 添加作业状态
        for job in state_dict.get('jobs', []):
            state_vector.append(job.get('completed', 0.0))
            state_vector.append(job.get('remaining_operations', 0))
            state_vector.append(job.get('due_date', 0.0))
            state_vector.append(job.get('current_operation', -1))
        
        # 添加配送商状态
        for distributor in state_dict.get('distributors', []):
            state_vector.append(distributor.get('required_jobs', 0))
            state_vector.append(distributor.get('delivered_jobs', 0))
            state_vector.append(distributor.get('weight', 0.0))
        
        # 填充到固定长度
        while len(state_vector) < 20:
            state_vector.append(0.0)
        
        return np.array(state_vector[:20], dtype=np.float32)
    
    def _action_to_dict(self, action: int) -> Dict:
        """
        将PPO动作索引转换为环境期望的动作格式
        """
        action_dict = {
            'action_type': 'schedule',
            'job_id': action % 100,
            'machine_id': (action // 100) % 10,
            'operation_id': (action // 1000) % 10
        }
        return action_dict
    
    def _get_valid_actions(self) -> List[int]:
        """
        根据当前环境状态生成有效动作列表
        """
        valid_actions = []
        
        if not self.environment:
            return [19999]
        
        available_jobs = getattr(self.environment, 'available_jobs', [])
        available_machines = getattr(self.environment, 'machines', [])
        
        for job in available_jobs:
            for machine in available_machines:
                action = getattr(job, 'job_id', 0) * 100 + getattr(machine, 'machine_id', 0)
                if action < 20000:
                    valid_actions.append(action)
        
        if not valid_actions:
            valid_actions = [19999]
        
        return valid_actions
    
    def _collect_results(self, stats: Dict, instance_data: Dict, episodes: int) -> Dict:
        """收集和整理结果"""
        
        # 计算核心指标
        total_tardiness = 0.0
        for job in self.environment.jobs:
            due = float(job.due_date)
            completed = float(job.completed_time) if hasattr(job, 'completed_time') else 0.0
            total_tardiness += max(0.0, completed - due)
        
        weighted_shortage = 0.0
        for distributor in self.environment.distributors:
            required = int(distributor.required_jobs)
            weight = float(distributor.weight)
            delivered = sum(1 for job in self.environment.dispatched_jobs 
                          if job.distributor_id == distributor.distributor_id)
            shortage = max(0, required - delivered)
            weighted_shortage += weight * shortage
        
        objective_sum = total_tardiness + weighted_shortage
        
        # 辅助指标
        makespan = float(self.environment.current_time)
        
        on_time_rate = np.mean([
            1.0 if float(job.completed_time) <= float(job.due_date) else 0.0
            for job in self.environment.jobs
        ]) if self.environment.jobs else 0.0
        
        # 需求覆盖率
        cover_vals = []
        for distributor in self.environment.distributors:
            required = int(distributor.required_jobs)
            if required <= 0:
                continue
            delivered = sum(1 for job in self.environment.dispatched_jobs 
                          if job.distributor_id == distributor.distributor_id)
            cover_vals.append(min(1.0, delivered / required))
        req_coverage = float(np.mean(cover_vals)) if cover_vals else 0.0
        
        # 平均派遣延迟
        late_dispatch = np.mean([
            max(0.0, float(job.dispatched_time) - float(job.due_date))
            for job in self.environment.jobs
        ]) if self.environment.jobs else 0.0
        
        # 构建结果字典
        result = {
            'algorithm': self.name,
            'instance_data': {
                'num_jobs': len(self.environment.jobs),
                'num_machines': len(self.environment.machines),
                'num_distributors': len(self.environment.distributors)
            },
            'performance_metrics': {
                'total_tardiness': float(total_tardiness),
                'weighted_shortage': float(weighted_shortage),
                'objective_sum': float(objective_sum),
                'makespan': float(makespan),
                'on_time_rate': float(on_time_rate),
                'req_coverage': float(req_coverage),
                'late_dispatch': float(late_dispatch)
            },
            'training_stats': {
                'episodes': episodes,
                'avg_reward': float(np.mean(stats['episode_rewards'])),
                'avg_makespan': float(np.mean(stats['makespans'])),
                'avg_completed_jobs': float(np.mean(stats['completed_jobs'])),
                'avg_dispatched_jobs': float(np.mean(stats['dispatched_jobs']))
            },
            'schedule': self._extract_schedule(),
            'dispatch_plan': self._extract_dispatch_plan()
        }
        
        return result
    
    def _extract_schedule(self) -> List[Dict]:
        """提取调度计划"""
        schedule = []
        for machine in self.environment.machines:
            for operation in machine.operations:
                schedule.append({
                    'machine_id': machine.machine_id,
                    'job_id': operation.job_id,
                    'operation_id': operation.operation_id,
                    'start_time': operation.start_time,
                    'end_time': operation.end_time,
                    'processing_time': operation.processing_time
                })
        return schedule
    
    def _extract_dispatch_plan(self) -> List[Dict]:
        """提取配送计划"""
        dispatch_plan = []
        for job in self.environment.dispatched_jobs:
            dispatch_plan.append({
                'job_id': job.job_id,
                'distributor_id': job.distributor_id,
                'dispatch_time': job.dispatched_time,
                'due_date': job.due_date
            })
        return dispatch_plan
    
    def save_model(self, filepath: str):
        """保存训练好的模型"""
        if self.agent:
            self.agent.save_model(filepath)
    
    def load_model(self, filepath: str):
        """加载训练好的模型"""
        if self.agent:
            self.agent.load_model(filepath)
