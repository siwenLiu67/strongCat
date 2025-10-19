"""
DQN智能体模块
基于深度Q网络的强化学习智能体，用于集成调度与配送问题
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
from typing import List, Tuple, Optional

from ..data_structures import EnvironmentState
from ..config import Config


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
    
    def __init__(self, state_dim: int, action_dim: int, config: Config):
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
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_net(state_tensor)
                
                # 只考虑有效动作
                valid_q_values = []
                for action in valid_actions:
                    if action < self.action_dim:
                        valid_q_values.append((q_values[0][action].item(), action))
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
        self.memory.push(state, action_idx, reward, next_state, done)
    
    def update(self):
        """更新网络"""
        if len(self.memory) < self.batch_size:
            return 0
        
        # 采样批次
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
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
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'q_net_state_dict': self.q_net.state_dict(),
            'target_q_net_state_dict': self.target_q_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps': self.steps,
            'epsilon': self.epsilon
        }, filepath)
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath)
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_q_net.load_state_dict(checkpoint['target_q_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps = checkpoint['steps']
        self.epsilon = checkpoint['epsilon']
