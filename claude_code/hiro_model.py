"""
HIRO (Hierarchical Reinforcement Learning with Off-Policy Correction) 
用于求解集成柔性作业车间调度与派遣问题 (IFJSSP-DP)

HIRO算法特点:
1. 分层架构：高层策略制定子目标，低层策略执行具体动作
2. Off-Policy修正：解决高低层策略更新频率不匹配问题
3. 目标条件：高层为低层提供有意义的子目标指导
4. 经验回放：分别维护高层和低层的经验池
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import copy
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time

from case_generator import FlexibleJobShopScenario
from config import Config
from data_structures import Job, Operation, Machine, Distributor


@dataclass
class HIROConfig:
    """HIRO算法配置"""
    # 网络参数
    high_level_dim: int = 128
    low_level_dim: int = 64
    goal_dim: int = 32
    
    # 训练参数
    batch_size: int = 64
    learning_rate_high: float = 3e-4
    learning_rate_low: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005  # 软更新参数
    
    # HIRO特定参数
    c: int = 10  # 高层策略更新频率
    horizon: int = 10  # 子目标时间跨度
    goal_threshold: float = 1.0  # 目标达成阈值
    
    # 经验回放
    buffer_size: int = 100000
    min_buffer_size: int = 1000
    
    # 探索参数
    noise_scale: float = 0.1
    exploration_episodes: int = 50


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int, goal_dim: int = 0):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        
        self.states = np.zeros((capacity, state_dim))
        self.actions = np.zeros((capacity, action_dim))
        self.rewards = np.zeros((capacity, 1))
        self.next_states = np.zeros((capacity, state_dim))
        self.dones = np.zeros((capacity, 1))
        
        if goal_dim > 0:
            self.goals = np.zeros((capacity, goal_dim))
            self.achieved_goals = np.zeros((capacity, goal_dim))
            self.next_achieved_goals = np.zeros((capacity, goal_dim))
        
        self.ptr = 0
        self.size = 0
    
    def add(self, state, action, reward, next_state, done, goal=None, achieved_goal=None, next_achieved_goal=None):
        """添加经验"""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        if self.goal_dim > 0:
            self.goals[self.ptr] = goal if goal is not None else np.zeros(self.goal_dim)
            self.achieved_goals[self.ptr] = achieved_goal if achieved_goal is not None else np.zeros(self.goal_dim)
            self.next_achieved_goals[self.ptr] = next_achieved_goal if next_achieved_goal is not None else np.zeros(self.goal_dim)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int):
        """采样batch"""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        batch = {
            'states': torch.FloatTensor(self.states[indices]),
            'actions': torch.FloatTensor(self.actions[indices]),
            'rewards': torch.FloatTensor(self.rewards[indices]),
            'next_states': torch.FloatTensor(self.next_states[indices]),
            'dones': torch.FloatTensor(self.dones[indices])
        }
        
        if self.goal_dim > 0:
            batch.update({
                'goals': torch.FloatTensor(self.goals[indices]),
                'achieved_goals': torch.FloatTensor(self.achieved_goals[indices]),
                'next_achieved_goals': torch.FloatTensor(self.next_achieved_goals[indices])
            })
        
        return batch


class HighLevelPolicy(nn.Module):
    """高层策略网络 - 生成子目标"""
    
    def __init__(self, state_dim: int, goal_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 目标生成器
        self.goal_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, goal_dim),
            nn.Tanh()  # 限制目标范围
        )
        
        # 价值函数
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        """前向传播"""
        encoded = self.state_encoder(state)
        goal = self.goal_generator(encoded)
        value = self.value_head(encoded)
        return goal, value


class LowLevelPolicy(nn.Module):
    """低层策略网络 - 执行具体动作"""
    
    def __init__(self, state_dim: int, goal_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        
        # 状态-目标融合网络
        self.fusion_net = nn.Sequential(
            nn.Linear(state_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor网络
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        # Critic网络
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, goal):
        """前向传播"""
        # 融合状态和目标
        fused = self.fusion_net(torch.cat([state, goal], dim=-1))
        action = self.actor(fused)
        return action
    
    def get_q_value(self, state, goal, action):
        """获取Q值"""
        fused = self.fusion_net(torch.cat([state, goal], dim=-1))
        q_input = torch.cat([fused, action], dim=-1)
        return self.critic(q_input)


class ScheduleEnvironment:
    """调度环境 - 将FJSP-DP转换为RL环境"""
    
    def __init__(self, scenario: FlexibleJobShopScenario):
        self.scenario = scenario
        self.jobs = scenario.jobs
        self.machines = scenario.machines
        self.distributors = scenario.distributors
        
        # 状态空间维度
        self.state_dim = self._calculate_state_dim()
        self.action_dim = self._calculate_action_dim()
        self.goal_dim = 32  # 子目标维度
        
        self.reset()
    
    def _calculate_state_dim(self) -> int:
        """计算状态空间维度"""
        # 作业状态 + 机器状态 + 时间信息 + 派遣状态
        job_features = len(self.jobs) * 4  # 进度、剩余时间、延误、完成状态
        machine_features = len(self.machines) * 3  # 忙碌状态、负载、可用时间
        time_features = 3  # 当前时间、平均延误、完工进度
        dispatch_features = len(self.distributors) * 2  # 批次数量、等待作业数
        
        return job_features + machine_features + time_features + dispatch_features
    
    def _calculate_action_dim(self) -> int:
        """计算动作空间维度"""
        # 调度动作 + 派遣动作
        schedule_actions = 3  # 作业选择概率、机器选择概率、优先级权重
        dispatch_actions = 2  # 批次组建决策、派遣时机决策
        
        return schedule_actions + dispatch_actions
    
    def reset(self):
        """重置环境"""
        self.current_time = 0
        self.scheduled_operations = []
        self.completed_jobs = set()
        self.machine_busy_until = {m.machine_id: 0 for m in self.machines}
        self.job_completion_times = {}
        self.dispatch_batches = []
        self.total_reward = 0
        
        # 初始化待调度工序
        self.ready_operations = []
        for job in self.jobs:
            if job.operations:
                self.ready_operations.append((job.job_id, 0))
        
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """获取当前状态"""
        state = []
        
        # 作业状态特征
        for job in self.jobs:
            if job.job_id in self.completed_jobs:
                state.extend([1.0, 0.0, 0.0, 1.0])  # 已完成
            else:
                # 计算进度
                progress = sum(1 for op_idx in range(len(job.operations)) 
                             if any(s['job_id'] == job.job_id and s['operation'] == op_idx 
                                   for s in self.scheduled_operations)) / len(job.operations)
                
                # 剩余处理时间估计
                remaining_ops = len(job.operations) - int(progress * len(job.operations))
                avg_proc_time = np.mean([min(op.processing_times.values()) 
                                       for op in job.operations]) if job.operations else 0
                remaining_time = remaining_ops * avg_proc_time
                
                # 延误计算
                due_date = getattr(job, 'due_date', 1000)
                tardiness = max(0, self.current_time - due_date) / 100.0  # 归一化
                
                state.extend([progress, remaining_time / 100.0, tardiness, 0.0])
        
        # 机器状态特征
        for machine in self.machines:
            busy = 1.0 if self.machine_busy_until[machine.machine_id] > self.current_time else 0.0
            load = len([s for s in self.scheduled_operations 
                       if s['machine'] == machine.machine_id]) / max(len(self.jobs), 1)
            available_time = max(0, self.machine_busy_until[machine.machine_id] - self.current_time) / 100.0
            
            state.extend([busy, load, available_time])
        
        # 时间特征
        time_progress = self.current_time / 200.0  # 归一化时间
        avg_tardiness = np.mean([max(0, self.current_time - getattr(job, 'due_date', 1000)) 
                                for job in self.jobs]) / 100.0
        completion_rate = len(self.completed_jobs) / len(self.jobs)
        
        state.extend([time_progress, avg_tardiness, completion_rate])
        
        # 派遣状态特征
        for distributor in self.distributors:
            dist_jobs = [job for job in self.jobs 
                        if getattr(job, 'distributor_id', 0) == distributor.distributor_id]
            batch_count = len([b for b in self.dispatch_batches 
                              if b['distributor'] == distributor.distributor_id])
            waiting_jobs = len([job for job in dist_jobs if job.job_id in self.completed_jobs])
            
            state.extend([batch_count / 10.0, waiting_jobs / max(len(dist_jobs), 1)])
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作"""
        action = np.clip(action, -1, 1)  # 确保动作在有效范围内
        
        # 解析动作
        schedule_action = action[:3]
        dispatch_action = action[3:5]
        
        reward = 0
        info = {}
        
        # 执行调度动作
        if self.ready_operations:
            schedule_reward = self._execute_schedule_action(schedule_action)
            reward += schedule_reward
        
        # 执行派遣动作
        if self.completed_jobs:
            dispatch_reward = self._execute_dispatch_action(dispatch_action)
            reward += dispatch_reward
        
        # 更新时间
        self.current_time += 1
        
        # 检查是否结束
        done = len(self.completed_jobs) == len(self.jobs) and len(self.ready_operations) == 0
        
        # 如果结束，计算最终奖励
        if done:
            final_reward = self._calculate_final_reward()
            reward += final_reward
            info['final_metrics'] = self._get_final_metrics()
        
        self.total_reward += reward
        next_state = self.get_state()
        
        return next_state, reward, done, info
    
    def _execute_schedule_action(self, action: np.ndarray) -> float:
        """执行调度动作"""
        if not self.ready_operations:
            return 0
        
        # 选择作业和工序
        job_prob = (action[0] + 1) / 2  # 转换到[0,1]
        machine_prob = (action[1] + 1) / 2
        priority_weight = action[2]
        
        # 计算作业优先级
        job_priorities = []
        for job_id, op_idx in self.ready_operations:
            job = next(j for j in self.jobs if j.job_id == job_id)
            due_date = getattr(job, 'due_date', 1000)
            
            # 多因素优先级计算
            urgency = max(0, due_date - self.current_time) / 100.0
            progress = op_idx / len(job.operations) if job.operations else 0
            
            priority = urgency * (1 - priority_weight) + progress * priority_weight
            job_priorities.append((job_id, op_idx, priority))
        
        # 选择最高优先级的作业
        if job_prob < 0.5:  # 贪婪选择
            selected = max(job_priorities, key=lambda x: x[2])
        else:  # 随机选择
            selected = random.choice(job_priorities)
        
        job_id, op_idx = selected[0], selected[1]
        
        # 选择机器
        job = next(j for j in self.jobs if j.job_id == job_id)
        operation = job.operations[op_idx]
        eligible_machines = getattr(operation, 'available_machine_ids', [])
        processing_times = getattr(operation, 'processing_times', {})
        
        if not eligible_machines or not processing_times:
            return -1  # 惩罚无效动作
        
        # 计算机器选择
        machine_scores = []
        for machine_id in eligible_machines:
            if machine_id in processing_times:
                proc_time = processing_times[machine_id]
                available_time = self.machine_busy_until[machine_id]
                
                # 机器评分：处理时间 + 可用时间
                score = 1.0 / (proc_time + max(0, available_time - self.current_time) + 1)
                machine_scores.append((machine_id, score))
        
        if not machine_scores:
            return -1
        
        if machine_prob < 0.5:  # 选择最佳机器
            selected_machine = max(machine_scores, key=lambda x: x[1])[0]
        else:  # 随机选择
            selected_machine = random.choice(machine_scores)[0]
        
        # 执行调度
        proc_time = processing_times[selected_machine]
        start_time = max(self.current_time, self.machine_busy_until[selected_machine])
        end_time = start_time + proc_time
        
        self.scheduled_operations.append({
            'job_id': job_id,
            'operation': op_idx,
            'machine': selected_machine,
            'start_time': start_time,
            'end_time': end_time,
            'processing_time': proc_time
        })
        
        self.machine_busy_until[selected_machine] = end_time
        self.ready_operations.remove((job_id, op_idx))
        
        # 检查是否有后续工序
        if op_idx + 1 < len(job.operations):
            self.ready_operations.append((job_id, op_idx + 1))
        else:
            # 作业完成
            self.completed_jobs.add(job_id)
            self.job_completion_times[job_id] = end_time
        
        # 计算调度奖励
        due_date = getattr(job, 'due_date', 1000)
        tardiness = max(0, end_time - due_date)
        schedule_reward = -tardiness / 100.0 - proc_time / 100.0  # 最小化延误和处理时间
        
        return schedule_reward
    
    def _execute_dispatch_action(self, action: np.ndarray) -> float:
        """执行派遣动作"""
        if not self.completed_jobs:
            return 0
        
        batch_threshold = (action[0] + 1) / 2  # 批次组建阈值
        dispatch_urgency = (action[1] + 1) / 2  # 派遣紧急度
        
        # 检查是否需要创建新批次
        if dispatch_urgency > 0.7 or len(self.completed_jobs) >= 3:
            return self._create_dispatch_batch(batch_threshold)
        
        return 0
    
    def _create_dispatch_batch(self, threshold: float) -> float:
        """创建派遣批次"""
        if not self.completed_jobs:
            return 0
        
        # 按配送商分组已完成作业
        distributor_jobs = {}
        for job_id in self.completed_jobs:
            job = next(j for j in self.jobs if j.job_id == job_id)
            dist_id = getattr(job, 'distributor_id', 0)
            
            if dist_id not in distributor_jobs:
                distributor_jobs[dist_id] = []
            
            completion_time = self.job_completion_times.get(job_id, self.current_time)
            distributor_jobs[dist_id].append((job_id, completion_time))
        
        total_reward = 0
        
        # 为每个配送商创建批次
        for dist_id, jobs in distributor_jobs.items():
            if len(jobs) >= 2 or threshold > 0.8:  # 批次大小判断
                # 创建批次
                job_ids = [job_id for job_id, _ in jobs]
                dispatch_time = max(comp_time for _, comp_time in jobs)
                
                self.dispatch_batches.append({
                    'distributor': dist_id,
                    'jobs': job_ids,
                    'dispatch_time': dispatch_time
                })
                
                # 计算派遣奖励
                batch_size_bonus = len(job_ids) * 0.1  # 批次大小奖励
                timeliness_bonus = sum(-max(0, dispatch_time - getattr(
                    next(j for j in self.jobs if j.job_id == job_id), 'due_date', 1000
                )) / 100.0 for job_id in job_ids)
                
                total_reward += batch_size_bonus + timeliness_bonus
                
                # 从已完成作业中移除已派遣的作业
                for job_id in job_ids:
                    self.completed_jobs.discard(job_id)
        
        return total_reward
    
    def _calculate_final_reward(self) -> float:
        """计算最终奖励"""
        # 总延误时间
        total_tardiness = 0
        for job in self.jobs:
            completion_time = self.job_completion_times.get(job.job_id, self.current_time)
            due_date = getattr(job, 'due_date', 1000)
            total_tardiness += max(0, completion_time - due_date)
        
        # 总完工时间
        makespan = max(self.job_completion_times.values()) if self.job_completion_times else self.current_time
        
        # 派遣效率
        total_dispatch_time = sum(batch['dispatch_time'] for batch in self.dispatch_batches)
        
        # 综合奖励（负值，因为要最小化）
        final_reward = -(total_tardiness / 100.0 + makespan / 100.0 + total_dispatch_time / 100.0)
        
        return final_reward
    
    def _get_final_metrics(self) -> Dict:
        """获取最终指标"""
        total_tardiness = sum(max(0, self.job_completion_times.get(job.job_id, self.current_time) - 
                                 getattr(job, 'due_date', 1000)) for job in self.jobs)
        makespan = max(self.job_completion_times.values()) if self.job_completion_times else self.current_time
        total_dispatch_time = sum(batch['dispatch_time'] for batch in self.dispatch_batches)
        
        return {
            'total_tardiness': total_tardiness,
            'makespan': makespan,
            'total_dispatch_time': total_dispatch_time,
            'total_reward': self.total_reward,
            'num_batches': len(self.dispatch_batches)
        }
    
    def get_achieved_goal(self) -> np.ndarray:
        """获取当前达成的目标"""
        # 基于当前状态计算达成的目标
        state = self.get_state()
        
        # 提取关键指标作为达成目标
        completion_rate = len(self.completed_jobs) / len(self.jobs)
        avg_machine_util = np.mean([len([s for s in self.scheduled_operations 
                                       if s['machine'] == m.machine_id]) 
                                  for m in self.machines]) / max(len(self.jobs), 1)
        
        avg_tardiness = np.mean([max(0, self.current_time - getattr(job, 'due_date', 1000)) 
                               for job in self.jobs]) / 100.0
        
        dispatch_efficiency = len(self.dispatch_batches) / max(len(self.completed_jobs), 1) if self.completed_jobs else 0
        
        # 组合成目标向量
        achieved_goal = np.array([
            completion_rate,
            avg_machine_util,
            -avg_tardiness,  # 负值因为我们希望延误更小
            dispatch_efficiency
        ] + [0] * (self.goal_dim - 4), dtype=np.float32)  # 填充到目标维度
        
        return achieved_goal


class HIROAgent:
    """HIRO智能体"""
    
    def __init__(self, env: ScheduleEnvironment, config: HIROConfig):
        self.env = env
        self.config = config
        
        # 网络初始化
        self.high_policy = HighLevelPolicy(env.state_dim, env.goal_dim, config.high_level_dim)
        self.high_policy_target = copy.deepcopy(self.high_policy)
        
        self.low_policy = LowLevelPolicy(env.state_dim, env.goal_dim, env.action_dim, config.low_level_dim)
        self.low_policy_target = copy.deepcopy(self.low_policy)
        
        # 优化器
        self.high_optimizer = optim.Adam(self.high_policy.parameters(), lr=config.learning_rate_high)
        self.low_optimizer = optim.Adam(self.low_policy.parameters(), lr=config.learning_rate_low)
        
        # 经验回放
        self.high_buffer = ReplayBuffer(config.buffer_size, env.state_dim, env.goal_dim, env.goal_dim)
        self.low_buffer = ReplayBuffer(config.buffer_size, env.state_dim + env.goal_dim, env.action_dim)
        
        # 训练统计
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_metrics = {
            'high_loss': [],
            'low_loss': [],
            'goal_achievement': []
        }
    
    def intrinsic_reward(self, state: np.ndarray, goal: np.ndarray, next_state: np.ndarray) -> float:
        """计算内在奖励"""
        # 计算状态变化
        achieved_goal = self.env.get_achieved_goal()
        
        # 目标导向的内在奖励
        goal_distance = np.linalg.norm(achieved_goal - goal)
        intrinsic_r = -goal_distance
        
        # 添加探索奖励
        exploration_bonus = 0.01 * np.random.random()
        
        return float(intrinsic_r + exploration_bonus)
    
    def off_policy_correction(self, trajectory: List, goal: np.ndarray) -> np.ndarray:
        """Off-policy修正"""
        # 重新标记目标以提高样本效率
        if len(trajectory) == 0:
            return goal
        
        # 使用轨迹终点状态作为修正目标
        final_achieved = trajectory[-1]['next_achieved_goal']
        
        # 插值修正
        corrected_goal = 0.8 * goal + 0.2 * final_achieved
        
        return corrected_goal
    
    def select_action(self, state: np.ndarray, goal: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        goal_tensor = torch.FloatTensor(goal).unsqueeze(0)
        
        with torch.no_grad():
            action = self.low_policy(state_tensor, goal_tensor).squeeze(0).numpy()
        
        if add_noise:
            noise = np.random.normal(0, self.config.noise_scale, size=action.shape)
            action = np.clip(action + noise, -1, 1)
        
        return action
    
    def select_goal(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """选择子目标"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            goal, _ = self.high_policy(state_tensor)
            goal = goal.squeeze(0).numpy()
        
        if add_noise:
            noise = np.random.normal(0, self.config.noise_scale, size=goal.shape)
            goal = np.clip(goal + noise, -1, 1)
        
        return goal
    
    def train_episode(self) -> Dict:
        """训练一个episode"""
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        # 高层轨迹
        high_trajectory = []
        current_goal = self.select_goal(state)
        goal_start_state = state.copy()
        goal_step = 0
        
        while True:
            # 低层执行
            action = self.select_action(state, current_goal)
            next_state, env_reward, done, info = self.env.step(action)
            
            # 计算内在奖励
            intrinsic_r = self.intrinsic_reward(state, current_goal, next_state)
            
            # 获取达成目标
            achieved_goal = self.env.get_achieved_goal()
            next_achieved_goal = self.env.get_achieved_goal()  # 更新后的达成目标
            
            # 存储低层经验
            low_state = np.concatenate([state, current_goal])
            low_next_state = np.concatenate([next_state, current_goal])
            
            self.low_buffer.add(
                low_state, action, intrinsic_r, low_next_state, done
            )
            
            episode_reward += env_reward
            episode_length += 1
            goal_step += 1
            
            # 高层决策更新
            if goal_step >= self.config.horizon or done:
                # 计算高层奖励
                high_reward = env_reward  # 可以设计更复杂的高层奖励
                
                # 存储高层经验
                self.high_buffer.add(
                    goal_start_state, current_goal, high_reward, next_state, done,
                    current_goal, achieved_goal, next_achieved_goal
                )
                
                high_trajectory.append({
                    'state': goal_start_state,
                    'goal': current_goal,
                    'reward': high_reward,
                    'next_state': next_state,
                    'achieved_goal': achieved_goal,
                    'next_achieved_goal': next_achieved_goal,
                    'done': done
                })
                
                # 选择新目标
                if not done:
                    current_goal = self.select_goal(next_state)
                    goal_start_state = next_state.copy()
                    goal_step = 0
            
            state = next_state
            
            if done:
                break
        
        # Off-policy修正
        if len(high_trajectory) > 1:
            corrected_goal = self.off_policy_correction(high_trajectory, high_trajectory[0]['goal'])
            # 重新存储修正后的经验（简化实现）
        
        # 训练网络
        training_info = {}
        if (self.high_buffer.size >= self.config.min_buffer_size and 
            self.low_buffer.size >= self.config.min_buffer_size):
            
            high_loss = self.train_high_level()
            low_loss = self.train_low_level()
            
            training_info = {
                'high_loss': high_loss,
                'low_loss': low_loss
            }
        
        # 记录统计信息
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        result = {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'final_metrics': info.get('final_metrics', {}),
            **training_info
        }
        
        return result
    
    def train_high_level(self) -> float:
        """训练高层策略"""
        if self.high_buffer.size < self.config.batch_size:
            return 0
        
        batch = self.high_buffer.sample(self.config.batch_size)
        
        states = batch['states']
        goals = batch['goals']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        
        # 计算目标Q值
        with torch.no_grad():
            next_goals, next_values = self.high_policy_target(next_states)
            target_q = rewards + self.config.gamma * (1 - dones) * next_values
        
        # 当前Q值
        current_goals, current_values = self.high_policy(states)
        
        # 计算损失
        value_loss = F.mse_loss(current_values, target_q)
        
        # 策略损失（简化实现）
        policy_loss = -current_values.mean()
        
        total_loss = value_loss + 0.1 * policy_loss
        
        # 更新网络
        self.high_optimizer.zero_grad()
        total_loss.backward()
        self.high_optimizer.step()
        
        # 软更新目标网络
        self.soft_update(self.high_policy, self.high_policy_target)
        
        return total_loss.item()
    
    def train_low_level(self) -> float:
        """训练低层策略"""
        if self.low_buffer.size < self.config.batch_size:
            return 0
        
        batch = self.low_buffer.sample(self.config.batch_size)
        
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        
        # 分离状态和目标
        batch_size = states.shape[0]
        state_dim = self.env.state_dim
        
        obs_states = states[:, :state_dim]
        goals = states[:, state_dim:]
        next_obs_states = next_states[:, :state_dim]
        next_goals = next_states[:, state_dim:]
        
        # 计算目标Q值
        with torch.no_grad():
            next_actions = self.low_policy_target(next_obs_states, next_goals)
            target_q = self.low_policy_target.get_q_value(next_obs_states, next_goals, next_actions)
            target_q = rewards + self.config.gamma * (1 - dones) * target_q
        
        # 当前Q值
        current_q = self.low_policy.get_q_value(obs_states, goals, actions)
        
        # Critic损失
        critic_loss = F.mse_loss(current_q, target_q)
        
        # Actor损失
        predicted_actions = self.low_policy(obs_states, goals)
        actor_loss = -self.low_policy.get_q_value(obs_states, goals, predicted_actions).mean()
        
        # 更新Critic
        self.low_optimizer.zero_grad()
        critic_loss.backward()
        self.low_optimizer.step()
        
        # 更新Actor
        self.low_optimizer.zero_grad()
        actor_loss.backward()
        self.low_optimizer.step()
        
        # 软更新目标网络
        self.soft_update(self.low_policy, self.low_policy_target)
        
        return (critic_loss + actor_loss).item()
    
    def soft_update(self, source: nn.Module, target: nn.Module):
        """软更新目标网络"""
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.config.tau * source_param.data + (1 - self.config.tau) * target_param.data
            )
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'high_policy': self.high_policy.state_dict(),
            'low_policy': self.low_policy.state_dict(),
            'config': self.config,
            'training_metrics': self.training_metrics
        }, filepath)
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath)
        self.high_policy.load_state_dict(checkpoint['high_policy'])
        self.low_policy.load_state_dict(checkpoint['low_policy'])
        self.training_metrics = checkpoint.get('training_metrics', self.training_metrics)


class HIROSolver:
    """HIRO求解器主类"""
    
    def __init__(self, scenario: FlexibleJobShopScenario, config: Optional[HIROConfig] = None):
        self.scenario = scenario
        self.config = config or HIROConfig()
        
        # 创建环境和智能体
        self.env = ScheduleEnvironment(scenario)
        self.agent = HIROAgent(self.env, self.config)
        
        # 训练历史
        self.training_history = []
    
    def train(self, num_episodes: int = 1000, save_interval: int = 100) -> Dict:
        """训练HIRO算法"""
        print(f"开始HIRO训练，共{num_episodes}个episode...")
        
        start_time = time.time()
        best_reward = float('-inf')
        
        for episode in range(num_episodes):
            episode_info = self.agent.train_episode()
            self.training_history.append(episode_info)
            
            episode_reward = episode_info['episode_reward']
            
            if episode_reward > best_reward:
                best_reward = episode_reward
            
            # 打印训练进度
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean([h['episode_reward'] for h in self.training_history[-10:]])
                avg_length = np.mean([h['episode_length'] for h in self.training_history[-10:]])
                
                print(f"Episode {episode + 1:4d}: "
                      f"Avg Reward = {avg_reward:8.2f}, "
                      f"Avg Length = {avg_length:6.1f}, "
                      f"Best = {best_reward:8.2f}")
            
            # 保存模型
            if (episode + 1) % save_interval == 0:
                self.agent.save_model(f'hiro_model_episode_{episode + 1}.pth')
        
        total_time = time.time() - start_time
        print(f"\n训练完成！总耗时: {total_time:.2f}秒")
        
        return {
            'total_episodes': num_episodes,
            'best_reward': best_reward,
            'final_avg_reward': np.mean([h['episode_reward'] for h in self.training_history[-100:]]),
            'training_time': total_time,
            'training_history': self.training_history
        }
    
    def evaluate(self, num_episodes: int = 10) -> Dict:
        """评估训练好的模型"""
        print(f"评估HIRO模型，共{num_episodes}个episode...")
        
        eval_results = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            current_goal = self.agent.select_goal(state, add_noise=False)
            goal_step = 0
            
            while True:
                action = self.agent.select_action(state, current_goal, add_noise=False)
                next_state, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
                goal_step += 1
                
                if goal_step >= self.config.horizon or done:
                    if not done:
                        current_goal = self.agent.select_goal(next_state, add_noise=False)
                        goal_step = 0
                
                state = next_state
                
                if done:
                    break
            
            eval_results.append({
                'episode_reward': episode_reward,
                'episode_length': episode_length,
                'final_metrics': info.get('final_metrics', {})
            })
            
            print(f"Eval Episode {episode + 1}: Reward = {episode_reward:.2f}")
        
        # 计算统计信息
        avg_reward = np.mean([r['episode_reward'] for r in eval_results])
        std_reward = np.std([r['episode_reward'] for r in eval_results])
        avg_length = np.mean([r['episode_length'] for r in eval_results])
        
        # 提取最终指标
        final_metrics = eval_results[-1]['final_metrics']
        
        print(f"\n评估结果:")
        print(f"平均奖励: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"平均长度: {avg_length:.1f}")
        if final_metrics:
            print(f"总延误: {final_metrics.get('total_tardiness', 0)}")
            print(f"完工时间: {final_metrics.get('makespan', 0)}")
            print(f"派遣时间: {final_metrics.get('total_dispatch_time', 0)}")
        
        return {
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'avg_length': avg_length,
            'eval_results': eval_results,
            'final_metrics': final_metrics
        }
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """绘制训练曲线"""
        if not self.training_history:
            print("没有训练历史数据")
            return
        
        episodes = range(1, len(self.training_history) + 1)
        rewards = [h['episode_reward'] for h in self.training_history]
        lengths = [h['episode_length'] for h in self.training_history]
        
        # 计算移动平均
        window = 50
        if len(rewards) >= window:
            moving_avg_rewards = []
            for i in range(window - 1, len(rewards)):
                moving_avg_rewards.append(np.mean(rewards[i - window + 1:i + 1]))
            
            plt.figure(figsize=(12, 8))
            
            # 奖励曲线
            plt.subplot(2, 2, 1)
            plt.plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Reward')
            plt.plot(episodes[window-1:], moving_avg_rewards, color='red', label=f'{window}-Episode Moving Average')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Training Rewards')
            plt.legend()
            plt.grid(True)
            
            # 长度曲线
            plt.subplot(2, 2, 2)
            plt.plot(episodes, lengths, alpha=0.7, color='green')
            plt.xlabel('Episode')
            plt.ylabel('Episode Length')
            plt.title('Episode Lengths')
            plt.grid(True)
            
            # 损失曲线（如果有的话）
            high_losses = [h.get('high_loss', 0) for h in self.training_history if 'high_loss' in h]
            low_losses = [h.get('low_loss', 0) for h in self.training_history if 'low_loss' in h]
            
            if high_losses:
                plt.subplot(2, 2, 3)
                plt.plot(high_losses, label='High-Level Loss', color='orange')
                plt.plot(low_losses, label='Low-Level Loss', color='purple')
                plt.xlabel('Training Step')
                plt.ylabel('Loss')
                plt.title('Training Losses')
                plt.legend()
                plt.grid(True)
            
            # 最终指标分布
            plt.subplot(2, 2, 4)
            recent_rewards = rewards[-100:] if len(rewards) >= 100 else rewards
            plt.hist(recent_rewards, bins=20, alpha=0.7, color='skyblue')
            plt.xlabel('Reward')
            plt.ylabel('Frequency')
            plt.title('Recent Reward Distribution')
            plt.grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"训练曲线已保存到: {save_path}")
            
            plt.show()


def main():
    """主函数"""
    print("HIRO (Hierarchical Reinforcement Learning) for FJSP-DP")
    print("=" * 60)
    
    # 创建问题实例
    config = Config()
    config.num_jobs = 6
    config.num_machines = 4
    config.num_distributors = 2
    config.min_operations = 2
    config.max_operations = 3
    
    scenario = FlexibleJobShopScenario(config=config)
    
    print(f"问题规模:")
    print(f"- 作业数量: {len(scenario.jobs)}")
    print(f"- 机器数量: {len(scenario.machines)}")
    print(f"- 配送商数量: {len(scenario.distributors)}")
    
    # 创建HIRO配置
    hiro_config = HIROConfig()
    hiro_config.batch_size = 32
    hiro_config.learning_rate_high = 1e-4
    hiro_config.learning_rate_low = 3e-4
    
    # 创建求解器
    solver = HIROSolver(scenario, hiro_config)
    
    # 训练模型
    training_results = solver.train(num_episodes=500, save_interval=100)
    
    print(f"\n训练结果:")
    print(f"最佳奖励: {training_results['best_reward']:.2f}")
    print(f"最终平均奖励: {training_results['final_avg_reward']:.2f}")
    print(f"训练时间: {training_results['training_time']:.2f}秒")
    
    # 评估模型
    eval_results = solver.evaluate(num_episodes=10)
    
    # 绘制训练曲线
    solver.plot_training_curves('hiro_training_curves.png')
    
    # 保存最终模型
    solver.agent.save_model('hiro_final_model.pth')
    
    print("\nHIRO算法求解完成！")


if __name__ == "__main__":
    main()