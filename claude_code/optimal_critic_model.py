"""
Option-Critic Architecture for FJSP-DP
集成柔性作业车间调度与派遣问题的选项-评论家算法

Option-Critic特点:
1. 学习选项层次结构：高级行为策略
2. 内在选项策略：每个选项有自己的策略
3. 终止函数：学习何时结束当前选项
4. 选项价值函数：评估选项的长期价值
5. 内在动机：为选项提供内在奖励
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import copy
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time

from case_generator import FlexibleJobShopScenario
from config import Config
from data_structures import Job, Operation, Machine, Distributor


@dataclass
class OptionCriticConfig:
    """Option-Critic算法配置"""
    # 网络参数
    hidden_dim: int = 128
    option_dim: int = 64
    num_options: int = 8  # 选项数量
    
    # 训练参数
    batch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    
    # Option-Critic特定参数
    termination_reg: float = 0.01  # 终止正则化
    entropy_weight: float = 0.01   # 熵权重
    beta_reg: float = 0.01         # 选项终止正则化
    
    # 经验回放
    buffer_size: int = 100000
    min_buffer_size: int = 1000
    
    # 探索参数
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # 选项相关
    max_option_length: int = 20    # 最大选项长度
    intrinsic_reward_scale: float = 0.1


class OptionCriticReplayBuffer:
    """Option-Critic专用经验回放缓冲区"""
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int, num_options: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_options = num_options
        
        # 基本转移
        self.states = np.zeros((capacity, state_dim))
        self.actions = np.zeros((capacity, action_dim))
        self.rewards = np.zeros((capacity, 1))
        self.next_states = np.zeros((capacity, state_dim))
        self.dones = np.zeros((capacity, 1))
        
        # 选项相关
        self.options = np.zeros((capacity, 1), dtype=np.int32)
        self.next_options = np.zeros((capacity, 1), dtype=np.int32)
        self.option_terminations = np.zeros((capacity, 1))
        self.option_lengths = np.zeros((capacity, 1))
        
        # 内在奖励
        self.intrinsic_rewards = np.zeros((capacity, 1))
        
        self.ptr = 0
        self.size = 0
    
    def add(self, state, action, reward, next_state, done, option, next_option, 
            option_terminated, option_length, intrinsic_reward=0):
        """添加经验"""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.options[self.ptr] = option
        self.next_options[self.ptr] = next_option
        self.option_terminations[self.ptr] = option_terminated
        self.option_lengths[self.ptr] = option_length
        self.intrinsic_rewards[self.ptr] = intrinsic_reward
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int):
        """采样batch"""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return {
            'states': torch.FloatTensor(self.states[indices]),
            'actions': torch.FloatTensor(self.actions[indices]),
            'rewards': torch.FloatTensor(self.rewards[indices]),
            'next_states': torch.FloatTensor(self.next_states[indices]),
            'dones': torch.FloatTensor(self.dones[indices]),
            'options': torch.LongTensor(self.options[indices]).squeeze(),
            'next_options': torch.LongTensor(self.next_options[indices]).squeeze(),
            'option_terminations': torch.FloatTensor(self.option_terminations[indices]),
            'option_lengths': torch.FloatTensor(self.option_lengths[indices]),
            'intrinsic_rewards': torch.FloatTensor(self.intrinsic_rewards[indices])
        }


class OptionCriticNetwork(nn.Module):
    """Option-Critic主网络"""
    
    def __init__(self, state_dim: int, action_dim: int, num_options: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_options = num_options
        self.hidden_dim = hidden_dim
        
        # 共享特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 选项价值函数 Q(s, ω)
        self.option_value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_options)
        )
        
        # 内在选项策略 π(a|s,ω)
        self.intra_option_policies = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Tanh()
            ) for _ in range(num_options)
        ])
        
        # 终止函数 β(s,ω)
        self.termination_functions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            ) for _ in range(num_options)
        ])
        
        # 状态价值函数 V(s)
        self.state_value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 选项选择策略 π_Ω(ω|s)
        self.option_policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_options),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        """前向传播"""
        features = self.feature_extractor(state)
        
        # 选项价值
        option_values = self.option_value(features)
        
        # 内在策略
        intra_policies = []
        for policy in self.intra_option_policies:
            intra_policies.append(policy(features))
        intra_policies = torch.stack(intra_policies, dim=1)  # [batch, num_options, action_dim]
        
        # 终止概率
        terminations = []
        for termination in self.termination_functions:
            terminations.append(termination(features))
        terminations = torch.stack(terminations, dim=1).squeeze(-1)  # [batch, num_options]
        
        # 状态价值
        state_values = self.state_value(features)
        
        # 选项策略
        option_probs = self.option_policy(features)
        
        return {
            'option_values': option_values,
            'intra_policies': intra_policies,
            'terminations': terminations,
            'state_values': state_values,
            'option_probs': option_probs,
            'features': features
        }
    
    def get_option_action(self, state, option, deterministic=False):
        """获取特定选项的动作"""
        with torch.no_grad():
            outputs = self.forward(state)
            action = outputs['intra_policies'][:, option]
            
            if not deterministic:
                # 添加探索噪声
                noise = torch.normal(0, 0.1, size=action.shape)
                action = torch.clamp(action + noise, -1, 1)
            
            return action
    
    def get_termination_prob(self, state, option):
        """获取选项终止概率"""
        with torch.no_grad():
            outputs = self.forward(state)
            return outputs['terminations'][:, option]
    
    def select_option(self, state, epsilon=0.0):
        """选择选项"""
        with torch.no_grad():
            outputs = self.forward(state)
            option_probs = outputs['option_probs']
            
            if random.random() < epsilon:
                # 随机选择
                option = random.randint(0, self.num_options - 1)
            else:
                # 根据概率选择
                option = torch.multinomial(option_probs, 1).item()
            
            return option


class ScheduleEnvironmentOC:
    """调度环境 - Option-Critic版本"""
    
    def __init__(self, scenario: FlexibleJobShopScenario):
        self.scenario = scenario
        self.jobs = scenario.jobs
        self.machines = scenario.machines
        self.distributors = scenario.distributors
        
        # 状态空间维度
        self.state_dim = self._calculate_state_dim()
        self.action_dim = self._calculate_action_dim()
        
        # 选项语义定义
        self.option_semantics = {
            0: "urgent_jobs_first",      # 紧急作业优先
            1: "shortest_job_first",     # 短作业优先  
            2: "load_balancing",         # 负载均衡
            3: "machine_utilization",    # 机器利用率优化
            4: "batch_optimization",     # 批次优化
            5: "early_dispatch",         # 早期派遣
            6: "quality_focus",          # 质量关注
            7: "exploration_mode"        # 探索模式
        }
        
        self.reset()
    
    def _calculate_state_dim(self) -> int:
        """计算状态空间维度"""
        job_features = len(self.jobs) * 6  # 增加更多作业特征
        machine_features = len(self.machines) * 4  # 增加机器特征
        time_features = 5  # 增加时间特征
        dispatch_features = len(self.distributors) * 3  # 增加派遣特征
        global_features = 8  # 全局特征
        
        return job_features + machine_features + time_features + dispatch_features + global_features
    
    def _calculate_action_dim(self) -> int:
        """计算动作空间维度"""
        schedule_actions = 4  # 作业排序、机器选择、优先级、时机
        dispatch_actions = 3  # 批次大小、派遣时机、配送商选择
        
        return schedule_actions + dispatch_actions
    
    def reset(self):
        """重置环境"""
        self.current_time = 0
        self.scheduled_operations = []
        self.completed_jobs = set()
        self.machine_busy_until = {m.machine_id: 0 for m in self.machines}
        self.job_completion_times = {}
        self.job_progress = {job.job_id: 0 for job in self.jobs}
        self.dispatch_batches = []
        self.total_reward = 0
        self.step_count = 0
        
        # 性能指标
        self.tardiness_penalty = 0
        self.makespan_penalty = 0
        self.dispatch_penalty = 0
        
        # 初始化待调度工序
        self.ready_operations = []
        for job in self.jobs:
            if job.operations:
                self.ready_operations.append((job.job_id, 0))
        
        # 选项上下文
        self.option_context = {
            'current_focus': 'schedule',  # schedule or dispatch
            'urgency_level': 0.0,
            'machine_utilization': 0.0,
            'dispatch_pressure': 0.0
        }
        
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """获取增强状态表示"""
        state = []
        
        # 作业状态特征
        for job in self.jobs:
            if job.job_id in self.completed_jobs:
                state.extend([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])  # 已完成
            else:
                progress = self.job_progress[job.job_id] / len(job.operations)
                remaining_ops = len(job.operations) - self.job_progress[job.job_id]
                avg_proc_time = np.mean([min(op.processing_times.values()) 
                                       for op in job.operations]) if job.operations else 0
                remaining_time = remaining_ops * avg_proc_time / 100.0
                
                due_date = getattr(job, 'due_date', 1000)
                urgency = max(0, due_date - self.current_time) / 100.0
                tardiness = max(0, self.current_time - due_date) / 100.0
                
                all_due_dates = [getattr(j, 'due_date', 1000) for j in self.jobs]
                relative_priority = (due_date - min(all_due_dates)) / max(max(all_due_dates) - min(all_due_dates), 1)
                
                state.extend([progress, remaining_time, urgency, tardiness, 0.0, relative_priority])
        
        # 机器状态特征
        for machine in self.machines:
            busy = 1.0 if self.machine_busy_until[machine.machine_id] > self.current_time else 0.0
            load = len([s for s in self.scheduled_operations 
                       if s['machine'] == machine.machine_id]) / max(len(self.jobs), 1)
            available_time = max(0, self.machine_busy_until[machine.machine_id] - self.current_time) / 100.0
            
            machine_ops = [s for s in self.scheduled_operations if s['machine'] == machine.machine_id]
            efficiency = 1.0 / (np.mean([op['processing_time'] for op in machine_ops]) + 1) if machine_ops else 0.5
            
            state.extend([busy, load, available_time, efficiency])
        
        # 时间特征
        time_progress = min(self.current_time / 200.0, 1.0)
        avg_tardiness = np.mean([max(0, self.current_time - getattr(job, 'due_date', 1000)) 
                               for job in self.jobs]) / 100.0
        completion_rate = len(self.completed_jobs) / len(self.jobs)
        workload_variance = np.var([self.machine_busy_until[m.machine_id] for m in self.machines]) / 10000.0
        
        total_operations = sum(len(job.operations) for job in self.jobs)
        scheduled_operations_count = len(self.scheduled_operations)
        schedule_progress = scheduled_operations_count / max(total_operations, 1)
        
        state.extend([time_progress, avg_tardiness, completion_rate, workload_variance, schedule_progress])
        
        # 派遣状态特征
        for distributor in self.distributors:
            dist_jobs = [job for job in self.jobs 
                        if getattr(job, 'distributor_id', 0) == distributor.distributor_id]
            completed_dist_jobs = [job for job in dist_jobs if job.job_id in self.completed_jobs]
            
            batch_count = len([b for b in self.dispatch_batches 
                              if b['distributor'] == distributor.distributor_id])
            waiting_jobs = len(completed_dist_jobs)
            
            if batch_count > 0:
                avg_batch_size = waiting_jobs / batch_count
                dispatch_efficiency = min(avg_batch_size / 3.0, 1.0)
            else:
                dispatch_efficiency = 0.0
            
            state.extend([batch_count / 10.0, waiting_jobs / max(len(dist_jobs), 1), dispatch_efficiency])
        
        # 全局特征
        global_features = [
            self.option_context['urgency_level'],
            self.option_context['machine_utilization'],
            self.option_context['dispatch_pressure'],
            len(self.ready_operations) / max(len(self.jobs), 1),
            self.step_count / 1000.0,
            min(self.total_reward / 100.0, 1.0),
            len(self.dispatch_batches) / 10.0,
            (self.tardiness_penalty + self.makespan_penalty + self.dispatch_penalty) / 300.0
        ]
        
        state.extend(global_features)
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action: np.ndarray, option: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作并返回选项相关信息"""
        action = np.clip(action, -1, 1)
        self.step_count += 1
        
        # 根据选项调整动作解释
        action = self._interpret_action_with_option(action, option)
        
        # 执行动作
        schedule_action = action[:4]
        dispatch_action = action[4:7]
        
        reward = 0
        intrinsic_reward = 0
        info: Dict[str, Any] = {'option_executed': option}
        
        # 执行调度动作
        if self.ready_operations:
            schedule_reward, schedule_intrinsic = self._execute_schedule_action_with_option(
                schedule_action, option)
            reward += schedule_reward
            intrinsic_reward += schedule_intrinsic
        
        # 执行派遣动作
        if self.completed_jobs:
            dispatch_reward, dispatch_intrinsic = self._execute_dispatch_action_with_option(
                dispatch_action, option)
            reward += dispatch_reward
            intrinsic_reward += dispatch_intrinsic
        
        # 更新时间和上下文
        self.current_time += 1
        self._update_option_context()
        
        # 选项特定的内在奖励
        option_intrinsic = self._calculate_option_intrinsic_reward(option)
        intrinsic_reward += option_intrinsic
        
        # 检查是否结束
        done = len(self.completed_jobs) == len(self.jobs) and len(self.ready_operations) == 0
        
        if done:
            final_reward = self._calculate_final_reward()
            reward += final_reward
            info['final_metrics'] = self._get_final_metrics()
        
        self.total_reward += reward
        next_state = self.get_state()
        
        # 添加选项相关信息
        info.update({
            'intrinsic_reward': float(intrinsic_reward),
            'option_context': dict(self.option_context.copy()),
            'option_semantics': str(self.option_semantics[option])
        })
        
        return next_state, reward, done, info
    
    def _interpret_action_with_option(self, action: np.ndarray, option: int) -> np.ndarray:
        """根据选项语义调整动作解释"""
        interpreted_action = action.copy()
        
        if option == 0:  # urgent_jobs_first
            interpreted_action[0] = abs(interpreted_action[0])
        elif option == 1:  # shortest_job_first
            interpreted_action[1] = -abs(interpreted_action[1])
        elif option == 2:  # load_balancing
            interpreted_action[2] = 0.0
        elif option == 3:  # machine_utilization
            interpreted_action[3] = abs(interpreted_action[3])
        elif option == 4:  # batch_optimization
            interpreted_action[4] = abs(interpreted_action[4])
        elif option == 5:  # early_dispatch
            interpreted_action[5] = abs(interpreted_action[5])
        elif option == 6:  # quality_focus
            interpreted_action[1] = abs(interpreted_action[1])
        elif option == 7:  # exploration_mode
            noise = np.random.normal(0, 0.2, size=interpreted_action.shape)
            interpreted_action = np.clip(interpreted_action + noise, -1, 1)
        
        return interpreted_action
    
    def _execute_schedule_action_with_option(self, action: np.ndarray, option: int) -> Tuple[float, float]:
        """根据选项执行调度动作"""
        if not self.ready_operations:
            return 0, 0
        
        job_selection_param = action[0]
        machine_selection_param = action[1]
        priority_weight = action[2]
        timing_param = action[3]
        
        # 根据选项计算作业优先级
        job_priorities = []
        for job_id, op_idx in self.ready_operations:
            job = next(j for j in self.jobs if j.job_id == job_id)
            due_date = getattr(job, 'due_date', 1000)
            operation = job.operations[op_idx]
            
            urgency = max(0, due_date - self.current_time)
            progress = op_idx / len(job.operations)
            
            if option == 0:  # urgent_jobs_first
                priority = 1000 - urgency
            elif option == 1:  # shortest_job_first
                avg_proc_time = np.mean(list(operation.processing_times.values()))
                priority = -avg_proc_time
            elif option == 6:  # quality_focus
                priority = len(operation.processing_times)
            else:
                priority = urgency * (1 - priority_weight) + progress * priority_weight
            
            job_priorities.append((job_id, op_idx, priority))
        
        # 选择作业
        if job_selection_param > 0:
            selected = max(job_priorities, key=lambda x: x[2])
        else:
            selected = random.choice(job_priorities)
        
        job_id, op_idx = selected[0], selected[1]
        
        # 选择机器
        job = next(j for j in self.jobs if j.job_id == job_id)
        operation = job.operations[op_idx]
        eligible_machines = getattr(operation, 'available_machine_ids', [])
        processing_times = getattr(operation, 'processing_times', {})
        
        if not eligible_machines or not processing_times:
            return -1, 0
        
        machine_scores = []
        for machine_id in eligible_machines:
            if machine_id in processing_times:
                proc_time = processing_times[machine_id]
                available_time = self.machine_busy_until[machine_id]
                
                if option == 2:  # load_balancing
                    load = len([s for s in self.scheduled_operations 
                               if s['machine'] == machine_id])
                    score = -load
                elif option == 3:  # machine_utilization
                    score = available_time
                else:
                    score = 1.0 / (proc_time + max(0, available_time - self.current_time) + 1)
                
                machine_scores.append((machine_id, score))
        
        if not machine_scores:
            return -1, 0
        
        if machine_selection_param > 0:
            selected_machine = max(machine_scores, key=lambda x: x[1])[0]
        else:
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
            'processing_time': proc_time,
            'option': option
        })
        
        self.machine_busy_until[selected_machine] = end_time
        self.ready_operations.remove((job_id, op_idx))
        self.job_progress[job_id] += 1
        
        # 检查作业是否完成
        if op_idx + 1 < len(job.operations):
            self.ready_operations.append((job_id, op_idx + 1))
        else:
            self.completed_jobs.add(job_id)
            self.job_completion_times[job_id] = float(end_time)
        
        # 计算奖励
        due_date = getattr(job, 'due_date', 1000)
        tardiness = max(0, end_time - due_date)
        
        schedule_reward = -tardiness / 100.0 - proc_time / 100.0
        
        # 选项特定奖励
        intrinsic_reward = 0
        if option == 0 and tardiness == 0:
            intrinsic_reward += 0.5
        elif option == 1 and proc_time < 5:
            intrinsic_reward += 0.3
        elif option == 2:
            machine_loads = [len([s for s in self.scheduled_operations 
                                 if s['machine'] == m.machine_id]) for m in self.machines]
            if max(machine_loads) - min(machine_loads) <= 1:
                intrinsic_reward += 0.4
        
        self.tardiness_penalty += tardiness
        
        return schedule_reward, intrinsic_reward
    
    def _execute_dispatch_action_with_option(self, action: np.ndarray, option: int) -> Tuple[float, float]:
        """根据选项执行派遣动作"""
        if not self.completed_jobs:
            return 0, 0
        
        batch_threshold = (action[0] + 1) / 2
        dispatch_urgency = (action[1] + 1) / 2
        distributor_preference = action[2]
        
        if option == 4:  # batch_optimization
            batch_threshold = max(0.8, batch_threshold)
        elif option == 5:  # early_dispatch
            dispatch_urgency = max(0.7, dispatch_urgency)
        
        if dispatch_urgency > 0.6 or len(self.completed_jobs) >= 3:
            return self._create_dispatch_batch_with_option(batch_threshold, option)
        
        return 0, 0
    
    def _create_dispatch_batch_with_option(self, threshold: float, option: int) -> Tuple[float, float]:
        """根据选项创建派遣批次"""
        if not self.completed_jobs:
            return 0, 0
        
        # 按配送商分组
        distributor_jobs = defaultdict(list)
        for job_id in self.completed_jobs:
            job = next(j for j in self.jobs if j.job_id == job_id)
            dist_id = getattr(job, 'distributor_id', 0)
            completion_time = self.job_completion_times.get(job_id, self.current_time)
            distributor_jobs[dist_id].append((job_id, completion_time))
        
        total_reward = 0
        total_intrinsic = 0
        
        for dist_id, jobs in distributor_jobs.items():
            if option == 4:  # batch_optimization
                min_batch_size = 3
            elif option == 5:  # early_dispatch
                min_batch_size = 1
            else:
                min_batch_size = 2
            
            if len(jobs) >= min_batch_size or threshold > 0.8:
                job_ids = [job_id for job_id, _ in jobs]
                dispatch_time = max(comp_time for _, comp_time in jobs)
                
                self.dispatch_batches.append({
                    'distributor': dist_id,
                    'jobs': job_ids,
                    'dispatch_time': dispatch_time,
                    'option': option
                })
                
                batch_size_bonus = len(job_ids) * 0.1
                timeliness_bonus = sum(-max(0, dispatch_time - getattr(
                    next(j for j in self.jobs if j.job_id == job_id), 'due_date', 1000
                )) / 100.0 for job_id in job_ids)
                
                total_reward += batch_size_bonus + timeliness_bonus
                
                if option == 4 and len(job_ids) >= 3:
                    total_intrinsic += 0.5
                elif option == 5:
                    avg_waiting_time = np.mean([dispatch_time - comp_time for _, comp_time in jobs])
                    if avg_waiting_time < 5:
                        total_intrinsic += 0.3
                
                for job_id in job_ids:
                    self.completed_jobs.discard(job_id)
        
        self.dispatch_penalty += total_reward
        
        return total_reward, total_intrinsic
    
    def _calculate_option_intrinsic_reward(self, option: int) -> float:
        """计算选项特定的内在奖励"""
        intrinsic = 0
        
        if option == 0:  # urgent_jobs_first
            current_tardiness = np.mean([max(0, self.current_time - getattr(job, 'due_date', 1000)) 
                                       for job in self.jobs])
            intrinsic = -current_tardiness / 100.0
        elif option == 2:  # load_balancing
            machine_loads = [self.machine_busy_until[m.machine_id] for m in self.machines]
            load_variance = np.var(machine_loads)
            intrinsic = -load_variance / 10000.0
        elif option == 3:  # machine_utilization
            total_busy_time = sum(self.machine_busy_until.values())
            total_time = len(self.machines) * max(self.current_time, 1)
            utilization = total_busy_time / total_time
            intrinsic = utilization * 0.1
        
        return float(intrinsic)
    
    def _update_option_context(self):
        """更新选项上下文"""
        avg_slack = np.mean([max(0, getattr(job, 'due_date', 1000) - self.current_time) for job in self.jobs])
        self.option_context['urgency_level'] = max(0.0, 1 - float(avg_slack) / 100.0)
        
        total_busy = sum(max(0, busy_time - self.current_time) 
                        for busy_time in self.machine_busy_until.values())
        total_capacity = len(self.machines) * 10
        self.option_context['machine_utilization'] = min(total_busy / max(total_capacity, 1), 1.0)
        
        self.option_context['dispatch_pressure'] = len(self.completed_jobs) / max(len(self.jobs), 1)
    
    def _calculate_final_reward(self) -> float:
        """计算最终奖励"""
        total_tardiness = sum(max(0, self.job_completion_times.get(job.job_id, self.current_time) - 
                                 getattr(job, 'due_date', 1000)) for job in self.jobs)
        makespan = max(self.job_completion_times.values()) if self.job_completion_times else self.current_time
        total_dispatch_time = sum(batch['dispatch_time'] for batch in self.dispatch_batches)
        
        final_reward = -(total_tardiness / 100.0 + makespan / 100.0 + total_dispatch_time / 100.0)
        return final_reward
    
    def _get_final_metrics(self) -> Dict:
        """获取最终指标"""
        total_tardiness = sum(max(0, self.job_completion_times.get(job.job_id, self.current_time) - 
                                 getattr(job, 'due_date', 1000)) for job in self.jobs)
        makespan = max(self.job_completion_times.values()) if self.job_completion_times else self.current_time
        total_dispatch_time = sum(batch['dispatch_time'] for batch in self.dispatch_batches)
        
        option_usage = defaultdict(int)
        for op in self.scheduled_operations:
            option_usage[op.get('option', -1)] += 1
        
        return {
            'total_tardiness': total_tardiness,
            'makespan': makespan,
            'total_dispatch_time': total_dispatch_time,
            'total_reward': self.total_reward,
            'num_batches': len(self.dispatch_batches),
            'option_usage': dict(option_usage),
            'option_context_final': self.option_context.copy()
        }


class OptionCriticAgent:
    """Option-Critic智能体"""
    
    def __init__(self, env: ScheduleEnvironmentOC, config: OptionCriticConfig):
        self.env = env
        self.config = config
        
        # 网络初始化
        self.network = OptionCriticNetwork(
            env.state_dim, env.action_dim, config.num_options, config.hidden_dim
        )
        self.target_network = copy.deepcopy(self.network)
        
        # 优化器
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.learning_rate)
        
        # 经验回放
        self.replay_buffer = OptionCriticReplayBuffer(
            config.buffer_size, env.state_dim, env.action_dim, config.num_options
        )
        
        # 训练状态
        self.epsilon = config.epsilon_start
        self.current_option = None
        self.option_length = 0
        
        # 统计信息
        self.training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'option_usage': defaultdict(int),
            'termination_rates': defaultdict(list),
            'intrinsic_rewards': []
        }
    
    def select_option(self, state: np.ndarray, force_new: bool = False) -> int:
        """选择选项"""
        if self.current_option is None or force_new:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            option = self.network.select_option(state_tensor, self.epsilon)
            self.current_option = option
            self.option_length = 0
            self.training_metrics['option_usage'][option] += 1
        
        return int(self.current_option)
    
    def should_terminate_option(self, state: np.ndarray) -> bool:
        """判断是否应该终止当前选项"""
        if self.current_option is None:
            return True
            
        # 强制终止条件
        if self.option_length >= self.config.max_option_length:
            return True
        
        try:
            # 学习的终止条件
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            termination_prob = self.network.get_termination_prob(state_tensor, self.current_option)
            
            should_terminate = random.random() < termination_prob.item()
            
            # 记录终止率
            self.training_metrics['termination_rates'][self.current_option].append(termination_prob.item())
        
            return should_terminate
        except Exception as e:
            print(f"终止检查出错: {e}")
            return True  # 出错时终止当前选项
    
    def get_action(self, state: np.ndarray, option: int) -> np.ndarray:
        """获取当前选项的动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.network.get_option_action(state_tensor, option, deterministic=False)
        return action.squeeze(0).numpy()
    
    def train_step(self):
        """执行一步训练"""
        if self.replay_buffer.size < self.config.min_buffer_size:
            return {}
        
        # 采样批次
        batch = self.replay_buffer.sample(self.config.batch_size)
        
        # 计算损失
        losses = self._compute_losses(batch)
        
        # 反向传播
        total_loss = (losses['option_value_loss'] + 
                     losses['intra_policy_loss'] + 
                     losses['termination_loss'])
        
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        
        self.optimizer.step()
        
        # 软更新目标网络
        self._soft_update_target_network()
        
        # 更新epsilon
        self.epsilon = max(self.config.epsilon_end, 
                          self.epsilon * self.config.epsilon_decay)
        
        return {k: v.item() for k, v in losses.items()}
    
    def _compute_losses(self, batch):
        """计算各种损失"""
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        options = batch['options']
        next_options = batch['next_options']
        option_terminations = batch['option_terminations']
        intrinsic_rewards = batch['intrinsic_rewards']
        
        # 当前网络输出
        current_outputs = self.network(states)
        current_option_values = current_outputs['option_values']
        current_intra_policies = current_outputs['intra_policies']
        current_terminations = current_outputs['terminations']
        
        # 目标网络输出
        with torch.no_grad():
            next_outputs = self.target_network(next_states)
            next_option_values = next_outputs['option_values']
            next_terminations = next_outputs['terminations']
        
        # 选项价值损失 (Intra-Option Q-Learning)
        batch_indices = torch.arange(states.size(0))
        current_q_values = current_option_values[batch_indices, options]
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = next_option_values[batch_indices, next_options]
            next_term_probs = next_terminations[batch_indices, next_options]
            
            # 如果选项继续，使用同选项的Q值；如果终止，使用最大Q值
            continued_values = next_q_values
            terminated_values = torch.max(next_option_values, dim=1)[0]
            next_values = (1 - next_term_probs) * continued_values + next_term_probs * terminated_values
            
            target_q_values = rewards.squeeze() + self.config.gamma * next_values * (1 - dones.squeeze())
        
        option_value_loss = F.mse_loss(current_q_values, target_q_values)
        
        # 内在策略损失 (Policy Gradient)
        selected_actions = current_intra_policies[batch_indices, options]
        action_log_probs = -0.5 * ((selected_actions - actions).pow(2)).sum(dim=1)
        
        # 优势函数
        with torch.no_grad():
            advantages = current_q_values - torch.mean(current_option_values, dim=1)
        
        intra_policy_loss = -(action_log_probs * advantages.detach()).mean()
        
        # 终止函数损失
        current_term_probs = current_terminations[batch_indices, options]
        
        # 终止优势：比较继续当前选项vs选择最佳选项
        with torch.no_grad():
            option_advantages = current_q_values - torch.max(current_option_values, dim=1)[0]
            termination_targets = (option_advantages < 0).float()
        
        termination_loss = F.binary_cross_entropy(current_term_probs, termination_targets)
        
        # 添加正则化
        termination_loss += self.config.termination_reg * current_term_probs.mean()
        
        return {
            'option_value_loss': option_value_loss,
            'intra_policy_loss': intra_policy_loss,
            'termination_loss': termination_loss
        }
    
    def _soft_update_target_network(self):
        """软更新目标网络"""
        for target_param, param in zip(self.target_network.parameters(), 
                                     self.network.parameters()):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )
    
    def train_episode(self):
        """训练一个episode"""
        state = self.env.reset()
        total_reward = 0
        total_intrinsic_reward = 0
        episode_length = 0
        option_switches = 0
        
        # 选择初始选项
        option = self.select_option(state, force_new=True)
        option_trajectory = []
        current_option_length = 0
        
        # 添加初始选项到轨迹
        option_trajectory.append({
            'option': option,
            'start_step': episode_length,
            'length': 0
        })
        
        max_steps = 1000  # 防止无限循环
        
        while episode_length < max_steps:
            # 只在非第一步检查选项终止
            if episode_length > 0 and self.should_terminate_option(state):
                # 更新当前选项长度
                if option_trajectory:
                    option_trajectory[-1]['length'] = current_option_length
                
                # 选择新选项
                option = self.select_option(state, force_new=True)
                option_switches += 1
                current_option_length = 0
                
                # 添加新选项到轨迹
                option_trajectory.append({
                    'option': option,
                    'start_step': episode_length,
                    'length': 0
                })
            
            # 获取动作并执行
            try:
                action = self.get_action(state, option)
                next_state, reward, done, info = self.env.step(action, option)
            except Exception as e:
                print(f"环境step出错: {e}")
                break
            
            intrinsic_reward = info.get('intrinsic_reward', 0)
            total_reward += reward
            total_intrinsic_reward += intrinsic_reward
            episode_length += 1
            current_option_length += 1
            self.option_length += 1
            
            # 存储经验
            next_option = option
            option_terminated = False
            
            if not done:
                if self.should_terminate_option(next_state):
                    next_option = self.select_option(next_state, force_new=True)
                    option_terminated = True
            
            self.replay_buffer.add(
                state, action, reward, next_state, done,
                option, next_option, option_terminated, 
                self.option_length, intrinsic_reward
            )
            
            # 训练
            if self.replay_buffer.size >= self.config.min_buffer_size:
                try:
                    losses = self.train_step()
                except Exception as e:
                    print(f"训练步骤出错: {e}")
            
            if done:
                if option_trajectory:
                    option_trajectory[-1]['length'] = current_option_length
                break
            
            state = next_state
            if option_terminated:
                option = next_option
                current_option_length = 0
                self.option_length = 0
            
            # 每100步输出一次调试信息
            if episode_length % 100 == 0:
                print(f"Episode步数: {episode_length}, 当前选项: {option}, 奖励: {reward:.2f}")
        
        if episode_length >= max_steps:
            print(f"Episode达到最大步数限制 ({max_steps})，强制结束")
        
        # 重置状态
        self.current_option = None
        self.option_length = 0
        
        # 记录统计信息
        self.training_metrics['episode_rewards'].append(total_reward)
        self.training_metrics['episode_lengths'].append(episode_length)
        self.training_metrics['intrinsic_rewards'].append(total_intrinsic_reward)
        
        return {
            'episode_reward': total_reward,
            'episode_length': episode_length,
            'intrinsic_reward': total_intrinsic_reward,
            'option_switches': option_switches,
            'option_trajectory': option_trajectory,
            'final_metrics': info.get('final_metrics', {})
        }
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_metrics': self.training_metrics,
            'epsilon': self.epsilon
        }, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"模型已从 {filepath} 加载")


class OptionCriticSolver:
    """Option-Critic求解器"""
    
    def __init__(self, scenario: FlexibleJobShopScenario, config: OptionCriticConfig):
        self.scenario = scenario
        self.config = config
        self.env = ScheduleEnvironmentOC(scenario)
        self.agent = OptionCriticAgent(self.env, config)
        self.training_history = []
    
    def train(self, num_episodes: int = 1000, save_interval: int = 100):
        """训练Option-Critic算法"""
        print(f"开始训练Option-Critic算法，共{num_episodes}轮...")
        
        best_reward = float('-inf')
        start_time = time.time()
        
        for episode in range(num_episodes):
            
            print(f"=== 训练进度: {episode + 1}/{num_episodes} ===")
            print(f"当前epsilon: {self.agent.epsilon:.3f}")

            episode_info = self.agent.train_episode()
            self.training_history.append(episode_info)
            
            episode_reward = episode_info['episode_reward']
            if episode_reward > best_reward:
                best_reward = episode_reward
            
            # 打印进度
            if (episode + 1) % 50 == 0:
                recent_rewards = [info['episode_reward'] for info in self.training_history[-50:]]
                recent_intrinsic = [info['intrinsic_reward'] for info in self.training_history[-50:]]
                recent_switches = [info['option_switches'] for info in self.training_history[-50:]]
                
                print(f"Episode {episode + 1}/{num_episodes}")
                print(f"  平均奖励: {np.mean(recent_rewards):.2f}")
                print(f"  平均内在奖励: {np.mean(recent_intrinsic):.2f}")
                print(f"  平均选项切换: {np.mean(recent_switches):.1f}")
                print(f"  当前epsilon: {self.agent.epsilon:.3f}")
                print(f"  最佳奖励: {best_reward:.2f}")
                
                # 选项使用统计
                option_counts = self.agent.training_metrics['option_usage']
                print(f"  选项使用分布: {dict(option_counts)}")
            
            # 保存模型
            if (episode + 1) % save_interval == 0:
                model_path = f"option_critic_model_episode_{episode + 1}.pth"
                self.agent.save_model(model_path)
        
        training_time = time.time() - start_time
        print(f"\n训练完成！总耗时: {training_time:.2f}秒")
        
        # 计算最终平均奖励
        final_rewards = [info['episode_reward'] for info in self.training_history[-100:]]
        final_avg_reward = np.mean(final_rewards)
        
        return {
            'best_reward': best_reward,
            'final_avg_reward': final_avg_reward,
            'training_time': training_time,
            'total_episodes': num_episodes
        }
    
    def evaluate(self, num_episodes: int = 10) -> Dict:
        """评估训练好的模型"""
        print(f"开始评估，共{num_episodes}轮...")
        
        eval_results = []
        self.agent.epsilon = 0.0  # 关闭探索
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            total_intrinsic_reward = 0
            episode_length = 0
            option_switches = 0
            
            # 选择初始选项
            option = self.agent.select_option(state, force_new=True)
            
            while True:
                # 检查选项终止
                if self.agent.should_terminate_option(state):
                    option = self.agent.select_option(state, force_new=True)
                    option_switches += 1
                
                # 执行动作
                action = self.agent.get_action(state, option)
                next_state, reward, done, info = self.env.step(action, option)
                
                intrinsic_reward = info.get('intrinsic_reward', 0)
                total_reward += reward
                total_intrinsic_reward += intrinsic_reward
                episode_length += 1
                
                if done:
                    break
                
                state = next_state
            
            eval_results.append({
                'reward': total_reward,
                'intrinsic_reward': total_intrinsic_reward,
                'length': episode_length,
                'switches': option_switches,
                'final_metrics': info.get('final_metrics', {})
            })
            
            print(f"  Episode {episode + 1}: 奖励={total_reward:.2f}, "
                  f"长度={episode_length}, 切换={option_switches}")
        
        # 重置状态
        self.agent.current_option = None
        self.agent.option_length = 0
        
        # 计算统计信息
        rewards = [r['reward'] for r in eval_results]
        intrinsic_rewards = [r['intrinsic_reward'] for r in eval_results]
        lengths = [r['length'] for r in eval_results]
        switches = [r['switches'] for r in eval_results]
        
        # 最终性能指标
        final_metrics = eval_results[-1]['final_metrics']
        
        return {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_intrinsic': np.mean(intrinsic_rewards),
            'avg_length': np.mean(lengths),
            'avg_switches': np.mean(switches),
            'eval_results': eval_results,
            'final_metrics': final_metrics
        }
    
    def analyze_options(self) -> Dict:
        """分析学到的选项"""
        print("=== 选项学习分析 ===")
        
        analysis = {}
        option_usage = self.agent.training_metrics['option_usage']
        termination_rates = self.agent.training_metrics['termination_rates']
        
        for option_id in range(self.config.num_options):
            semantic = self.env.option_semantics[option_id]
            usage_count = option_usage.get(option_id, 0)
            
            if option_id in termination_rates and termination_rates[option_id]:
                avg_termination = np.mean(termination_rates[option_id])
                std_termination = np.std(termination_rates[option_id])
            else:
                avg_termination = 0.0
                std_termination = 0.0
            
            print(f"选项 {option_id} ({semantic}):")
            print(f"  使用次数: {usage_count}")
            print(f"  平均终止率: {avg_termination:.3f} ± {std_termination:.3f}")
            
            analysis[option_id] = {
                'semantic': semantic,
                'usage_count': usage_count,
                'avg_termination_rate': avg_termination,
                'std_termination_rate': std_termination
            }
        
        return analysis
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """绘制训练曲线"""
        if not self.training_history:
            print("没有训练历史数据")
            return
        
        episodes = range(1, len(self.training_history) + 1)
        rewards = [info['episode_reward'] for info in self.training_history]
        intrinsic_rewards = [info['intrinsic_reward'] for info in self.training_history]
        lengths = [info['episode_length'] for info in self.training_history]
        
        # 计算移动平均
        window = min(50, len(rewards) // 10)
        if len(rewards) >= window:
            moving_avg_rewards = []
            moving_avg_intrinsic = []
            
            for i in range(window - 1, len(rewards)):
                moving_avg_rewards.append(np.mean(rewards[i - window + 1:i + 1]))
                moving_avg_intrinsic.append(np.mean(intrinsic_rewards[i - window + 1:i + 1]))
            
            plt.figure(figsize=(15, 10))
            
            # 奖励曲线
            plt.subplot(2, 3, 1)
            plt.plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Reward')
            plt.plot(episodes[window-1:], moving_avg_rewards, color='red', 
                    label=f'{window}-Episode Moving Average')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Training Rewards')
            plt.legend()
            plt.grid(True)
            
            # 内在奖励曲线
            plt.subplot(2, 3, 2)
            plt.plot(episodes, intrinsic_rewards, alpha=0.3, color='green', label='Intrinsic Reward')
            plt.plot(episodes[window-1:], moving_avg_intrinsic, color='orange', 
                    label=f'{window}-Episode Moving Average')
            plt.xlabel('Episode')
            plt.ylabel('Intrinsic Reward')
            plt.title('Intrinsic Rewards')
            plt.legend()
            plt.grid(True)
            
            # 长度曲线
            plt.subplot(2, 3, 3)
            plt.plot(episodes, lengths, alpha=0.5, color='purple')
            plt.xlabel('Episode')
            plt.ylabel('Episode Length')
            plt.title('Episode Lengths')
            plt.grid(True)
            
            # 选项使用频率
            plt.subplot(2, 3, 4)
            option_counts = [self.agent.training_metrics['option_usage'].get(i, 0) 
                           for i in range(self.config.num_options)]
            option_labels = [f"Option {i}\n({self.env.option_semantics.get(i, 'unknown')[:10]}...)" 
                           for i in range(self.config.num_options)]
            plt.bar(range(self.config.num_options), option_counts)
            plt.xlabel('Options')
            plt.ylabel('Usage Count')
            plt.title('Option Usage Distribution')
            plt.xticks(range(self.config.num_options), option_labels, rotation=45, fontsize=8)
            
            # 终止率分布
            plt.subplot(2, 3, 5)
            avg_termination_rates = []
            for i in range(self.config.num_options):
                rates = self.agent.training_metrics['termination_rates'].get(i, [])
                avg_rate = np.mean(rates) if rates else 0
                avg_termination_rates.append(avg_rate)
            
            plt.bar(range(self.config.num_options), avg_termination_rates, alpha=0.7, color='orange')
            plt.xlabel('Options')
            plt.ylabel('Average Termination Rate')
            plt.title('Option Termination Rates')
            plt.xticks(range(self.config.num_options), 
                      [f"Opt {i}" for i in range(self.config.num_options)])
            
            # 奖励分布
            plt.subplot(2, 3, 6)
            plt.hist(rewards[-200:], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            plt.xlabel('Episode Reward')
            plt.ylabel('Frequency')
            plt.title('Recent Reward Distribution')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"训练曲线已保存到: {save_path}")
            
            plt.show()
        else:
            print(f"数据不足以计算{window}期移动平均")
    
    def save_results(self, filepath: str):
        """保存完整结果"""
        results = {
            'config': self.config,
            'training_metrics': self.agent.training_metrics,
            'scenario_info': {
                'num_jobs': len(self.scenario.jobs),
                'num_machines': len(self.scenario.machines),
                'num_distributors': len(self.scenario.distributors)
            },
            'option_semantics': self.env.option_semantics,
            'final_analysis': self.analyze_options()
        }
        
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        print(f"结果已保存到: {filepath}")


def main():
    """主函数 - 完整演示Option-Critic算法"""
    print("=== Option-Critic Architecture for FJSP-DP ===")
    print("集成柔性作业车间调度与派遣问题的选项-评论家算法")
    print("="*60)
    
    # 创建问题实例
    config = Config()
    config.num_jobs = 6
    config.num_machines = 4
    config.num_distributors = 2
    config.min_operations = 2
    config.max_operations = 4
    
    print("\n=== 生成FJSP-DP实例 ===")
    scenario = FlexibleJobShopScenario(config=config)
    
    # 配置Option-Critic参数
    oc_config = OptionCriticConfig()
    oc_config.num_options = 8
    oc_config.hidden_dim = 128
    oc_config.learning_rate = 1e-3
    oc_config.batch_size = 64
    oc_config.max_option_length = 15
    oc_config.epsilon_start = 0.9
    oc_config.epsilon_end = 0.05
    oc_config.epsilon_decay = 0.995
    
    # 创建求解器
    solver = OptionCriticSolver(scenario, oc_config)
    
    print(f"\n=== 实例信息 ===")
    print(f"作业数量: {len(scenario.jobs)}")
    print(f"机器数量: {len(scenario.machines)}")
    print(f"配送商数量: {len(scenario.distributors)}")
    print(f"状态维度: {solver.env.state_dim}")
    print(f"动作维度: {solver.env.action_dim}")
    print(f"选项数量: {oc_config.num_options}")
    
    # 打印作业详情
    print(f"\n=== 作业详细信息 ===")
    for job in scenario.jobs:
        due_date = getattr(job, 'due_date', 'N/A')
        distributor_id = getattr(job, 'distributor_id', 'N/A')
        print(f"作业 {job.job_id}: {len(job.operations)}个工序, "
              f"截止时间={due_date}, 配送商={distributor_id}")
        for i, op in enumerate(job.operations):
            available_machines = getattr(op, 'available_machine_ids', [])
            processing_times = getattr(op, 'processing_times', {})
            print(f"  工序{i}: 机器{available_machines}, 时间{processing_times}")
    
    # 打印选项语义
    print(f"\n=== 选项语义定义 ===")
    for option_id, semantic in solver.env.option_semantics.items():
        print(f"选项 {option_id}: {semantic}")
    
    # 训练
    print(f"\n=== 开始训练 ===")
    num_episodes = 500
    training_results = solver.train(num_episodes=num_episodes, save_interval=100)
    
    print(f"\n=== 训练结果 ===")
    print(f"最佳奖励: {training_results['best_reward']:.2f}")
    print(f"最终平均奖励: {training_results['final_avg_reward']:.2f}")
    print(f"训练时间: {training_results['training_time']:.2f}秒")
    print(f"总训练轮数: {training_results['total_episodes']}")
    
    # 选项分析
    print(f"\n=== 选项学习分析 ===")
    analysis = solver.analyze_options()
    
    # 找出最常用和最少用的选项
    most_used = max(analysis.items(), key=lambda x: x[1]['usage_count'])
    least_used = min(analysis.items(), key=lambda x: x[1]['usage_count'])
    
    print(f"\n最常用选项: 选项{most_used[0]} ({most_used[1]['semantic']}) - 使用{most_used[1]['usage_count']}次")
    print(f"最少用选项: 选项{least_used[0]} ({least_used[1]['semantic']}) - 使用{least_used[1]['usage_count']}次")
    
    # 评估
    print(f"\n=== 开始评估 ===")
    eval_results = solver.evaluate(num_episodes=10)
    
    print(f"\n=== 评估结果 ===")
    print(f"平均奖励: {eval_results['avg_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"平均内在奖励: {eval_results['avg_intrinsic']:.2f}")
    print(f"平均轨迹长度: {eval_results['avg_length']:.1f}")
    print(f"平均选项切换: {eval_results['avg_switches']:.1f}")
    
    # 最终性能指标
    final_metrics = eval_results['final_metrics']
    if final_metrics:
        print(f"\n=== 最终性能指标 ===")
        print(f"总延误时间: {final_metrics.get('total_tardiness', 0)}")
        print(f"总完工时间: {final_metrics.get('makespan', 0)}")
        print(f"总派遣时间: {final_metrics.get('total_dispatch_time', 0)}")
        print(f"批次数量: {final_metrics.get('num_batches', 0)}")
        print(f"总目标值: {final_metrics.get('total_reward', 0):.2f}")
        
        option_usage_final = final_metrics.get('option_usage', {})
        print(f"最终选项使用分布: {option_usage_final}")
    
    # 绘制训练曲线
    print(f"\n=== 生成训练曲线 ===")
    solver.plot_training_curves('option_critic_training_curves.png')
    
    # 保存最终模型和结果
    print(f"\n=== 保存结果 ===")
    solver.agent.save_model('option_critic_final_model.pth')
    solver.save_results('option_critic_complete_results.pkl')
    
    print(f"\n=== 算法性能总结 ===")
    print(f"1. 训练效率: {num_episodes}轮训练用时{training_results['training_time']:.1f}秒")
    print(f"2. 学习效果: 最终平均奖励{eval_results['avg_reward']:.2f}")
    print(f"3. 选项多样性: {len([opt for opt in analysis.values() if opt['usage_count'] > 0])}个选项被使用")
    print(f"4. 决策复杂度: 平均每轨迹{eval_results['avg_switches']:.1f}次选项切换")
    
    return solver, training_results, eval_results


if __name__ == "__main__":
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("Option-Critic Architecture for FJSP-DP")
    print("集成柔性作业车间调度与派遣问题的选项-评论家算法")
    print("="*60)
    
    # 运行完整演示
    solver, training_results, eval_results = main()
    
    print("\n程序结束!")