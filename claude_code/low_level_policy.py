import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from collections import deque
import random

class LowLevelPolicy(nn.Module):
    """底层策略 - 负责执行具体的调度和配送动作"""
    
    def __init__(self, config, policy_type="scheduling"):
        super().__init__()
        self.config = config
        self.policy_type = policy_type
        
        # 根据策略类型设置不同的动作空间
        if policy_type == "scheduling":
            self.action_dim = 8  # 8种调度规则
            self.state_dim = 10  # 调度状态特征维度
        else:  # dispatching
            self.action_dim = 5  # 5种配送策略
            self.state_dim = 8   # 配送状态特征维度
            
        self.hidden_dim = 64
        
        # 网络结构
        self.fc1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.action_dim)
        
        # 目标网络
        self.target_net = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim)
        )
        self.target_net.load_state_dict(self.state_dict())
        
        # 训练参数
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update_freq = 100
        self.step_count = 0
        
        # 经验回放
        self.replay_buffer = deque(maxlen=5000)
        self.batch_size = 32
        
    def _extract_scheduling_features(self, state: Dict) -> torch.Tensor:
        """提取调度状态特征"""
        jobs = state['available_jobs']
        machines = state['machines']
        current_time = state.get('current_time', 0)
        
        # 作业相关特征
        waiting_jobs = [j for j in jobs if j.status == 'waiting']
        num_waiting_jobs = len(waiting_jobs)
        
        # 处理时间特征
        proc_times = []
        for job in waiting_jobs:
            if job.current_operation < len(job.operations):
                op = job.operations[job.current_operation]
                proc_times.extend(list(op.processing_times.values()))
        
        avg_proc_time = np.mean(proc_times) if proc_times else 0
        std_proc_time = np.std(proc_times) if proc_times else 0
        
        # 交期紧迫性
        due_dates = [getattr(j, 'due_date', float('inf')) for j in waiting_jobs]
        urgency = sum(1 for dd in due_dates if dd - current_time < 10) / max(1, len(due_dates))
        
        # 机器状态特征
        busy_machines = sum(1 for m in machines if m.status == 'busy')
        machine_utilization = busy_machines / len(machines) if machines else 0
        
        # 机器负载方差
        remaining_times = [m.remaining_time for m in machines]
        load_variance = np.var(remaining_times) if remaining_times else 0
        
        # 构建特征向量
        features = torch.tensor([
            num_waiting_jobs / max(1, len(jobs)),  # 等待作业比例
            avg_proc_time / max(1, self.config.max_processing_time),  # 归一化平均处理时间
            std_proc_time / max(1, self.config.max_processing_time),  # 归一化处理时间标准差
            urgency,                                # 交期紧迫性
            machine_utilization,                    # 机器利用率
            load_variance / max(1, self.config.max_processing_time),  # 归一化负载方差
            current_time / self.config.max_time_steps,  # 时间进度
            len([j for j in jobs if j.status == 'processing']) / max(1, len(jobs)),  # 加工中作业比例
            sum(getattr(j, 'waiting_time', 0) for j in waiting_jobs) / max(1, len(waiting_jobs) * 10),  # 平均等待时间
            busy_machines / max(1, len(machines))   # 忙碌机器比例
        ], dtype=torch.float32)
        
        return features.unsqueeze(0)
    
    def _extract_dispatching_features(self, state: Dict) -> torch.Tensor:
        """提取配送状态特征"""
        completed_jobs = state.get('completed_jobs', [])
        dispatched_jobs = state.get('dispatched_jobs', [])
        distributors = state.get('distributors', [])
        current_time = state.get('current_time', 0)
        
        # 已完成但未配送的作业
        undelivered_jobs = [j for j in completed_jobs if j.status == 'completed']
        num_undelivered = len(undelivered_jobs)
        
        # 配送紧迫性
        due_dates = [getattr(j, 'due_date', float('inf')) for j in undelivered_jobs]
        urgency = sum(1 for dd in due_dates if dd - current_time < 5) / max(1, len(due_dates))
        
        # 配送商需求
        total_requirements = 0
        unmet_requirements = 0
        
        for distributor in distributors:
            if hasattr(distributor, 'delivery_requirements') and distributor.delivery_requirements:
                requirement = distributor.delivery_requirements
                for due_time, ratio, weight in zip(requirement.due_times, requirement.ratios, requirement.weights):
                    if current_time <= due_time:
                        total_requirements += 1
                        completed_amount = sum(getattr(j, 'amount', 1) for j in undelivered_jobs 
                                             if getattr(j, 'distributor_id', -1) == distributor.distributor_id)
                        required_amount = ratio * getattr(distributor, 'total_amount', 1)
                        if completed_amount < required_amount:
                            unmet_requirements += 1
        
        requirement_pressure = unmet_requirements / max(1, total_requirements)
        
        # 批量配送效率
        batch_efficiency = len(dispatched_jobs) / max(1, len(completed_jobs) + len(dispatched_jobs))
        
        # 配送延迟
        avg_delay = np.mean([max(0, current_time - getattr(j, 'due_date', current_time)) 
                           for j in undelivered_jobs]) if undelivered_jobs else 0
        normalized_delay = avg_delay / max(1, self.config.max_time_steps)
        
        # 构建特征向量
        features = torch.tensor([
            num_undelivered / max(1, len(completed_jobs) + len(dispatched_jobs)),  # 未配送作业比例
            urgency,                                # 配送紧迫性
            requirement_pressure,                   # 需求压力
            batch_efficiency,                       # 批量配送效率
            normalized_delay,                       # 归一化配送延迟
            current_time / self.config.max_time_steps,  # 时间进度
            len(dispatched_jobs) / max(1, len(completed_jobs) + len(dispatched_jobs)),  # 已配送比例
            sum(getattr(j, 'amount', 1) for j in undelivered_jobs) / max(1, sum(getattr(j, 'amount', 1) for j in completed_jobs))  # 未配送量比例
        ], dtype=torch.float32)
        
        return features.unsqueeze(0)
    
    def _extract_state_features(self, state: Dict) -> torch.Tensor:
        """根据策略类型提取状态特征"""
        if self.policy_type == "scheduling":
            return self._extract_scheduling_features(state)
        else:
            return self._extract_dispatching_features(state)
    
    def select_action(self, state: Dict, subgoal: Dict = None) -> Tuple[int, Dict]:
        """选择底层动作"""
        state_features = self._extract_state_features(state)
        
        # epsilon-贪婪策略
        if random.random() < self.epsilon:
            action_idx = random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                q_values = self.forward(state_features)
                action_idx = q_values.argmax().item()
        
        # 更新epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # 根据策略类型执行具体动作
        if self.policy_type == "scheduling":
            action = self._execute_scheduling_action(action_idx, state, subgoal)
        else:
            action = self._execute_dispatching_action(action_idx, state, subgoal)
        
        return action_idx, action
    
    def _execute_scheduling_action(self, rule_idx: int, state: Dict, subgoal: Dict = None) -> Dict:
        """执行调度动作"""
        jobs = [j for j in state['available_jobs'] if j.status == 'waiting']
        machines = state['machines']
        
        if not jobs or not machines:
            return {'wait': True}
        
        # 根据规则对作业排序
        if rule_idx == 0:   # EDD - 最早交期优先
            jobs.sort(key=lambda j: getattr(j, 'due_date', float('inf')))
        elif rule_idx == 1: # SPT - 最短处理时间优先
            jobs.sort(key=lambda j: min(j.operations[j.current_operation].processing_times.values()))
        elif rule_idx == 2: # LPT - 最长处理时间优先
            jobs.sort(key=lambda j: -max(j.operations[j.current_operation].processing_times.values()))
        elif rule_idx == 3: # CR - 关键比率
            current_time = state.get('current_time', 0)
            jobs.sort(key=lambda j: (getattr(j, 'due_date', current_time) - current_time) / 
                     sum(min(op.processing_times.values()) for op in j.operations[j.current_operation:]))
        elif rule_idx == 4: # FCFS - 先到先服务
            jobs.sort(key=lambda j: getattr(j, 'arrival_time', 0))
        elif rule_idx == 5: # MWKR - 最多剩余工作
            jobs.sort(key=lambda j: -sum(min(op.processing_times.values()) 
                     for op in j.operations[j.current_operation:]))
        elif rule_idx == 6: # LWKR - 最少剩余工作
            jobs.sort(key=lambda j: sum(min(op.processing_times.values()) 
                     for op in j.operations[j.current_operation:]))
        else:              # Random - 随机
            random.shuffle(jobs)
        
        # 分配作业到空闲机器
        schedule = {}
        assigned_machines = set()
        
        for job in jobs:
            if job.current_operation >= len(job.operations):
                continue
                
            op = job.operations[job.current_operation]
            for m_id in op.available_machine_ids:
                machine = next((m for m in machines if m.machine_id == m_id), None)
                if machine and machine.status == 'waiting' and m_id not in assigned_machines:
                    schedule[job.job_id] = m_id
                    assigned_machines.add(m_id)
                    break
        
        if schedule:
            return {'schedule': schedule}
        else:
            return {'wait': True}
    
    def _execute_dispatching_action(self, strategy_idx: int, state: Dict, subgoal: Dict = None) -> Dict:
        """执行配送动作"""
        completed_jobs = state.get('completed_jobs', [])
        undelivered_jobs = [j for j in completed_jobs if j.status == 'completed']
        
        if not undelivered_jobs:
            return {'dispatch': {}}
        
        # 根据策略选择配送批次
        if strategy_idx == 0:   # 紧急优先
            jobs_to_dispatch = sorted(undelivered_jobs, 
                                    key=lambda j: getattr(j, 'due_date', float('inf')))[:2]
        elif strategy_idx == 1: # 批量配送
            jobs_to_dispatch = undelivered_jobs[:3]  # 前3个作业
        elif strategy_idx == 2: # 按配送商分组
            distributor_groups = {}
            for job in undelivered_jobs:
                distributor_id = getattr(job, 'distributor_id', 0)
                if distributor_id not in distributor_groups:
                    distributor_groups[distributor_id] = []
                distributor_groups[distributor_id].append(job)
            
            # 选择作业最多的配送商
            if distributor_groups:
                max_distributor = max(distributor_groups.keys(), 
                                    key=lambda k: len(distributor_groups[k]))
                jobs_to_dispatch = distributor_groups[max_distributor][:2]
            else:
                jobs_to_dispatch = []
        elif strategy_idx == 3: # 价值优先
            jobs_to_dispatch = sorted(undelivered_jobs, 
                                    key=lambda j: -getattr(j, 'amount', 1))[:2]
        else:                  # 随机选择
            jobs_to_dispatch = random.sample(undelivered_jobs, 
                                           min(2, len(undelivered_jobs)))
        
        if jobs_to_dispatch:
            batch_id = 1
            job_ids = [j.job_id for j in jobs_to_dispatch]
            return {'dispatch': {batch_id: job_ids}}
        else:
            return {'dispatch': {}}
    
    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = F.relu(self.fc1(state_features))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def update(self) -> float:
        """更新网络参数"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # 随机采样批次
        batch_samples = random.sample(self.replay_buffer, self.batch_size)
        
        states = torch.stack([sample[0] for sample in batch_samples])
        actions = torch.tensor([sample[1] for sample in batch_samples], dtype=torch.long)
        rewards = torch.tensor([sample[2] for sample in batch_samples], dtype=torch.float32)
        next_states = torch.stack([sample[3] for sample in batch_samples])
        dones = torch.tensor([sample[4] for sample in batch_samples], dtype=torch.float32)
        
        # 计算当前Q值
        current_q_values = self.forward(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # 计算损失
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.state_dict())
        
        return loss.item()
    
    def store_experience(self, state: Dict, action: int, reward: float, next_state: Dict, done: bool):
        """存储经验到回放缓冲区"""
        state_features = self._extract_state_features(state)
        next_state_features = self._extract_state_features(next_state)
        
        self.replay_buffer.append((
            state_features.squeeze(),
            action,
            reward,
            next_state_features.squeeze(),
            done
        ))
