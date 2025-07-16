import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List
from data_structures import Job, Machine

class RuleBasedDQNAgent:
    """基于规则和DQN的FJSP调度智能体"""
    
    def __init__(self, config):
        self.config = config
        self.state_dim = 10  # 状态维度
        self.action_dim = 8  # 8种调度规则
        
        # DQN网络
        self.q_net = DQNNetwork(self.state_dim, 128, self.action_dim)  # 增加网络宽度
        self.target_q_net = DQNNetwork(self.state_dim, 128, self.action_dim)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=0.0001)  # 降低学习率
        
        # 训练参数
        self.gamma = 0.99
        self.epsilon_start = 0.3  # 提高初始探索率
        self.epsilon_end = 0.01   # 降低最终探索率
        self.epsilon_decay = 10000 # 延长探索衰减周期
        self.epsilon = self.epsilon_start
        self.batch_size = 32
        self.target_update = 100  # 减少目标网络更新频率
        self.count = 0
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(10000)

    def _get_state(self, state: Dict) -> np.ndarray:
        """将环境状态转换为DQN输入状态"""
        jobs = state['available_jobs']
        machines = state['machines']
        
        # 提取关键特征
        avg_due_date = np.mean([j.due_date for j in jobs if hasattr(j, 'due_date')]) if jobs else 0
        proc_times = []
        if jobs:
            for j in jobs:
                for op in j.operations:
                    proc_times.extend(list(op.processing_times.values()))
        avg_proc_time = np.mean(proc_times) if proc_times else 0
        machine_util = np.mean([1 if m.status == 'busy' else 0 for m in machines]) if machines else 0
        
        # 安全获取状态字段，提供默认值
        current_time = state.get('current_time', 0)
        completed_jobs = state.get('completed_jobs', [])
        dispatched_jobs = state.get('dispatched_jobs', [])
        last_schedule_time = state.get('last_schedule_time', 0)
        last_batch_time = state.get('last_batch_time', 0)
        
        state_vec = np.array([
            len(jobs),
            len([j for j in jobs if j.status == 'waiting']),
            avg_due_date,
            avg_proc_time,
            machine_util,
            current_time / self.config.max_time_steps,
            len(completed_jobs),
            len(dispatched_jobs),
            last_schedule_time / self.config.max_time_steps,
            last_batch_time / self.config.max_time_steps
        ], dtype=np.float32)
        
        return state_vec

    def select_action(self, state: Dict):
        """选择调度动作
        返回格式: ({'schedule': {job_id: machine_id}}, rule_idx)
        """
        state_vec = self._get_state(state)
        
        # 更新epsilon值(线性衰减)
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                    np.exp(-1. * self.count / self.epsilon_decay)
        
        # epsilon-贪婪策略选择规则
        if np.random.random() < self.epsilon:
            rule_idx = np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                q_values = self.q_net(torch.FloatTensor(state_vec))
                rule_idx = q_values.argmax().item()
        
        # 保存最后使用的规则索引，以便在update时使用
        self.last_rule_idx = rule_idx
        
        # 应用选中的调度规则
        action = self._apply_rule(rule_idx, state)

        if action is None:
            action = {'wait': True}
        
        return action, rule_idx

    def _apply_rule(self, rule_idx: int, state: Dict) -> Dict[str, Dict[int, int]]:
        """应用指定的调度规则
        返回格式: {'schedule': {job_id: machine_id}}
        """
        jobs = [j for j in state['available_jobs'] if j.status == 'waiting']
        machines = state['machines']
        
        if not jobs or not machines:
            return {}
            
        # 根据规则对作业排序
        if rule_idx == 0:   # EDD
            jobs.sort(key=lambda j: j.due_date)
        elif rule_idx == 1: # SPT
            jobs.sort(key=lambda j: min(j.operations[j.current_operation].processing_times.values()))
        elif rule_idx == 2: # LPT
            jobs.sort(key=lambda j: -max(j.operations[j.current_operation].processing_times.values()))
        elif rule_idx == 3: # CR
            current_time = state.get('t', 0)
            jobs.sort(key=lambda j: (j.due_date - current_time) / 
                     sum(min(op.processing_times.values()) for op in j.operations[j.current_operation:]))
        elif rule_idx == 4: # FCFS
            jobs.sort(key=lambda j: j.job_id)
        elif rule_idx == 5: # MWKR
            jobs.sort(key=lambda j: -sum(min(op.processing_times.values()) 
                     for op in j.operations[j.current_operation:]))
        elif rule_idx == 6: # LWKR
            jobs.sort(key=lambda j: sum(min(op.processing_times.values()) 
                     for op in j.operations[j.current_operation:]))
        else:              # Random
            np.random.shuffle(jobs)
        
        # 分配作业到最早空闲机器
        schedule = {}
        for job in jobs:
            op = job.operations[job.current_operation]
            for m_id in op.available_machine_ids:
                if machines[m_id].status == 'waiting':
                    schedule[job.job_id] = m_id
                    break
        
        return {'schedule': schedule}

    def update(self, transition_dict: Dict):
        """更新DQN网络"""
        states = torch.FloatTensor([self._get_state(state) for state in transition_dict['states']])
        actions = torch.LongTensor(transition_dict['actions']).view(-1, 1)
        rewards = torch.FloatTensor(transition_dict['rewards']).view(-1, 1)
        next_states = torch.FloatTensor([self._get_state(state) for state in transition_dict['next_states']])
        dones = torch.FloatTensor(transition_dict['dones']).view(-1, 1)
        
        # 计算目标Q值
        with torch.no_grad():
            max_next_q = self.target_q_net(next_states).max(1)[0].view(-1, 1)
            q_targets = rewards + self.gamma * max_next_q * (1 - dones)
        
        # 计算当前Q值
        q_values = self.q_net(states).gather(1, actions)
        
        # 计算损失并更新
        loss = F.mse_loss(q_values, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1
        
        return loss.item()

class DQNNetwork(nn.Module):
    """DQN网络结构"""
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
        
    def __len__(self):
        return len(self.buffer)
