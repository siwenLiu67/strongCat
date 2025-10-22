import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from collections import deque
from typing import Dict, List, Tuple
from environment import WarehouseEnvironment

class DQNNetwork(nn.Module):
    """DQN网络 - 与现有DQN模型保持一致"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.FloatTensor(states), 
                torch.LongTensor(actions),
                torch.FloatTensor(rewards),
                torch.FloatTensor(next_states),
                torch.BoolTensor(dones))
    
    def __len__(self):
        return len(self.buffer)

class HighLevelDQNAgent:
    """高层DQN智能体 - 替换原有的策略梯度方法"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 网络参数
        self.state_dim = 9  # 与现有特征维度一致
        self.action_dim = 3  # 3种策略：生产、配送、等待
        
        # DQN网络
        self.q_net = DQNNetwork(self.state_dim, self.action_dim, hidden_dim=64)
        self.target_q_net = DQNNetwork(self.state_dim, self.action_dim, hidden_dim=64)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.learning_rate)
        
        # 经验回放
        self.memory = ReplayBuffer(capacity=1000)
        
        # 训练参数
        self.steps = 0
        self.epsilon = config.epsilon_start
        
        # 初始化目标网络
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        # 动作掩码相关
        self.last_action = None
        self.action_counts = torch.ones(1, 3)
        self.total_steps = 0

    def _build_features(self, state: Dict) -> torch.Tensor:
        """复用现有的特征构建方法"""
        jobs = state['available_jobs']
        completed_jobs = state.get('completed_jobs', [])
        dispatched_jobs = state.get('dispatched_jobs', [])
        machines = state['machines']
        t = state.get('current_time', 0)

        # 新特征
        total_jobs = len(jobs) + len(completed_jobs) + len(dispatched_jobs)
        completed_ratio = len(completed_jobs) / max(1, total_jobs)
        dispatched_ratio = len(dispatched_jobs) / max(1, total_jobs)
        avg_util = sum(getattr(m, 'utilization', 0.0) for m in machines) / max(1, len(machines))
        avg_wait = sum(getattr(j, 'waiting_time', 0.0) for j in jobs) / max(1, len(jobs))

        # 原有特征
        new_jobs = len(jobs) - len(completed_jobs)
        q_new = new_jobs / max(1, len(jobs))
        remaining_times = [float(m.remaining_time) for m in machines]
        sigma_mach_t = torch.tensor(remaining_times, dtype=torch.float32).std().item() / max(1e-6, self.config.max_processing_time)
        urgent_jobs = sum([1 for j in jobs if getattr(j, 'due_time', 1e9) <= t])
        urgency_ratio = urgent_jobs / len(jobs) if jobs else 0.0
        last_schedule_time = state.get('last_schedule_time', 0)
        schedule_age = (t - last_schedule_time) / max(1, self.config.max_processing_time)
        total_job_time = sum(getattr(j, 'processing_time', 0) for j in jobs)
        total_machine_capacity = sum(m.remaining_time for m in machines)
        system_pressure = total_job_time / max(1, total_machine_capacity)

        feats = torch.tensor([
            q_new,
            sigma_mach_t,
            urgency_ratio,
            schedule_age,
            system_pressure,
            completed_ratio,
            dispatched_ratio,
            avg_util,
            avg_wait
        ], dtype=torch.float32)
        
        # 检查特征值是否有效
        if torch.isnan(feats).any() or torch.isinf(feats).any():
            feats = torch.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)
            
        return feats.unsqueeze(0)

    def _get_action_mask(self, state: Dict) -> torch.Tensor:
        """复用现有的动作掩码方法"""
        mask = torch.ones(3, dtype=torch.bool)
        completed_jobs = state.get('completed_jobs', [])
        dispatched_jobs = state.get('dispatched_jobs', [])
        
        # 修复：检查是否有未配送的已完成作业
        completed_job_ids = {getattr(j, 'job_id', i) for i, j in enumerate(completed_jobs)}
        dispatched_job_ids = {getattr(j, 'job_id', i) for i, j in enumerate(dispatched_jobs)}
        
        # 如果没有未配送的已完成作业，禁用配送动作
        if not completed_job_ids or len(completed_job_ids - dispatched_job_ids) == 0:
            mask[1] = False

        # 如果都在配送中状态，禁用配送动作
        if all(getattr(j, 'status') in ['dispatching', 'dispatched'] for j in state['available_jobs']):
            mask[1] = False
       
        # 如果没有可调度作业，禁用调度动作
        if not state['available_jobs'] or not(getattr(j, 'status') in ['waiting'] for j in state['available_jobs']):
            mask[0] = False
        
        # 只要有空闲机器且有可调度作业，就允许schedule
        if not state['machines'] or all(m.remaining_time > 0 for m in state['machines']):
            mask[0] = False
            
        return mask.unsqueeze(0)

    def select_action(self, state: Dict) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """选择动作 - 使用DQN的ε-贪婪策略"""
        self.total_steps += 1
        
        # 构建特征
        feats = self._build_features(state)
        
        # 获取动作掩码
        mask = self._get_action_mask(state)
        
        # ε-贪婪策略
        if random.random() < self.epsilon:
            # 随机探索，但只选择有效动作
            valid_actions = [i for i, m in enumerate(mask.squeeze()) if m]
            if valid_actions:
                action = random.choice(valid_actions)
            else:
                action = 2  # 默认等待
        else:
            # 贪婪选择
            with torch.no_grad():
                q_values = self.q_net(feats)
                
                # 应用动作掩码
                masked_q_values = q_values.masked_fill(~mask, float('-inf'))
                
                # 选择Q值最大的动作
                action = masked_q_values.argmax().item()
        
        # 更新动作计数
        self.action_counts[0, action] += 1
        
        # 更新epsilon
        self.epsilon = max(self.config.epsilon_end, 
                          self.epsilon * self.config.epsilon_decay)
        
        # 返回动作和占位符（保持接口兼容）
        self.last_action = action
        return action, torch.tensor(0.0), torch.tensor(0.0)

    def store_experience(self, state, action, reward, next_state, done):
        """存储经验到回放缓冲区"""
        feats = self._build_features(state)
        next_feats = self._build_features(next_state)
        
        # 转换为numpy数组存储
        self.memory.push(
            feats.squeeze().detach().numpy(),
            action,
            reward,
            next_feats.squeeze().detach().numpy(),
            done
        )

    def update(self):
        """更新DQN网络"""
        if len(self.memory) < self.config.batch_size:
            return 0
        
        # 采样批次
        states, actions, rewards, next_states, dones = self.memory.sample(self.config.batch_size)
        
        # 计算当前Q值
        current_q_values = self.q_net(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_q_net(next_states).max(1)[0]
            target_q_values = rewards + (self.config.gamma * next_q_values * ~dones)
        
        # 计算损失
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # 更新目标网络
        if self.steps % self.config.target_update_frequency == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        self.steps += 1
        
        return loss.item()

    def parameters(self):
        """返回网络参数（保持接口兼容）"""
        return list(self.q_net.parameters())
