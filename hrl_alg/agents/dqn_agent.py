"""
深度Q网络(DQN)智能体实现
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
from typing import List, Dict, Any, Optional

class DQNNetwork(nn.Module):
    """深度Q网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
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

class DQNAgent:
    """DQN智能体"""
    
    def __init__(self, state_dim: int, action_dim: int, config: Any):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # 网络参数
        self.lr = 0.001
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 10000
        self.target_update_freq = 100
        self.batch_size = 32
        self.memory_size = 10000
        
        # 初始化神经网络
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_q_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        
        # 初始化经验回放
        self.memory = ReplayBuffer(self.memory_size)
        
        # 训练计数器
        self.steps = 0
        self.epsilon = self.epsilon_start
        
        # 更新目标网络
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def state_to_vector(self, state: Dict) -> np.ndarray:
        """将状态字典转换为向量表示
        
        Args:
            state: 包含状态信息的字典
            
        Returns:
            np.ndarray: 状态向量
        """
        state_vector = []
        
        # 全局特征
        state_vector.extend([
            state['current_time'],
            state['last_schedule_time'],
            state['last_batch_time'],
            len(state['available_jobs']),
            len(state['completed_jobs']),
            len(state['dispatched_jobs'])
        ])
        
        # 作业特征
        jobs = state['available_jobs']
        for job in jobs[:5]:  # 取前5个作业
            state_vector.extend([
                job.current_operation,
                len(job.operations),
                job.due_date,
                1.0 if job.status == 'waiting' else 0.0,
                1.0 if job.status == 'processing' else 0.0,
                1.0 if job.status == 'completed' else 0.0
            ])
            
        # 机器特征
        machines = state['machines']
        for machine in machines[:5]:  # 取前5个机器
            state_vector.extend([
                1.0 if machine.status == 'busy' else 0.0,
                machine.remaining_time,
                machine.total_busy_time / max(state['current_time'], 1)  # 利用率
            ])
            
        # 填充或截断到固定长度
        if len(state_vector) < self.state_dim:
            state_vector.extend([0] * (self.state_dim - len(state_vector)))
        else:
            state_vector = state_vector[:self.state_dim]
            
        return np.array(state_vector, dtype=np.float32)

    def action_to_index(self, action: Dict) -> int:
        """将动作字典转换为索引
        
        Args:
            action: 动作字典
            
        Returns:
            int: 动作索引
        """
        # 这里需要根据实际的动作结构来实现
        # 示例实现
        if action.get('wait', False):
            return self.action_dim - 1  # 等待动作
        
        job_id = action.get('job_id', 0)
        machine_id = action.get('machine_id', 0)
        return job_id * 100 + machine_id  # 简单的编码方式
        
        # 神经网络
        self.q_net = DQNNetwork(state_dim, action_dim)
        self.target_q_net = DQNNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        
        # 经验回放
        self.memory = ReplayBuffer(self.memory_size)
        
        # 训练计数器
        self.steps = 0
        self.epsilon = self.epsilon_start
        
        # 更新目标网络
        self.target_q_net.load_state_dict(self.q_net.state_dict())
    
    def select_action(self, state: np.ndarray, valid_actions: List[int]) -> int:
        """选择动作"""
        self.steps += 1
        
        # 更新epsilon
        self.epsilon = max(self.epsilon_end, 
                          self.epsilon_start - (self.epsilon_start - self.epsilon_end) * 
                          self.steps / self.epsilon_decay)
        
        if random.random() < self.epsilon:
            # 随机探索
            return random.choice(valid_actions)
        else:
            # 贪婪选择
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_net(state_tensor)
                
                # 只考虑有效动作
                valid_q_values = []
                for action in valid_actions:
                    if action < self.action_dim:
                        valid_q_values.append((q_values[0][action].cpu().item(), action))
                    else:
                        # 对于超出范围的动作使用默认值
                        valid_q_values.append((0.0, action))
                
                # 选择Q值最大的动作
                best_action = max(valid_q_values, key=lambda x: x[0])[1]
                return best_action
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储经验"""
        # 将动作映射到有效范围
        action_idx = min(action, self.action_dim - 1)
        # 确保数据都在CPU上
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
        self.memory.push(state, action_idx, reward, next_state, done)
    
    def update(self) -> Optional[float]:
        """更新网络"""
        if len(self.memory) < self.batch_size:
            return None
        
        # 采样批次
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # 将数据移到正确的设备上
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # 计算当前Q值
        current_q_values = self.q_net(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_q_net(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # 计算损失
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        if self.steps % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        return loss.item()