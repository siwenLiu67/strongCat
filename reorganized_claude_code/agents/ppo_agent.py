"""
PPO智能体模块
实现近端策略优化算法的智能体
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Tuple, Optional
from ..config import Config


class ActorNetwork(nn.Module):
    """Actor网络，输出动作概率分布"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(ActorNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class CriticNetwork(nn.Module):
    """Critic网络，输出状态价值"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super(CriticNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class PPOAgent:
    """
    PPO智能体类
    实现近端策略优化算法
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: Config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # 网络参数
        self.hidden_dim = config.get('ppo_hidden_dim', 128)
        self.lr = config.get('ppo_lr', 0.001)
        self.gamma = config.get('ppo_gamma', 0.99)
        self.epsilon = config.get('ppo_epsilon', 0.2)
        self.epochs = config.get('ppo_epochs', 10)
        self.batch_size = config.get('ppo_batch_size', 64)
        
        # 网络
        self.actor = ActorNetwork(state_dim, action_dim, self.hidden_dim)
        self.critic = CriticNetwork(state_dim, self.hidden_dim)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        
        # 经验缓冲区
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        
        # 训练统计
        self.episode_rewards = []
        
    def select_action(self, state: np.ndarray, valid_actions: List[int]) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        选择动作
        
        Args:
            state: 当前状态
            valid_actions: 有效动作列表
            
        Returns:
            action: 选择的动作
            log_prob: 动作的对数概率
            value: 状态价值
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # 获取动作概率分布
        action_probs = self.actor(state_tensor)
        
        # 创建有效动作掩码
        valid_mask = torch.zeros(self.action_dim)
        for action in valid_actions:
            if action < self.action_dim:
                valid_mask[action] = 1.0
        
        # 应用有效动作掩码
        masked_probs = action_probs * valid_mask
        if masked_probs.sum() == 0:
            # 如果没有有效动作，均匀分布
            masked_probs = valid_mask / valid_mask.sum()
        else:
            masked_probs = masked_probs / masked_probs.sum()
        
        # 从分布中采样动作
        dist = Categorical(masked_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # 获取状态价值
        value = self.critic(state_tensor)
        
        return action.item(), log_prob, value.squeeze()
    
    def store_transition(self, state: np.ndarray, action: int, log_prob: torch.Tensor, 
                        value: torch.Tensor, reward: float, done: bool):
        """
        存储经验转换
        
        Args:
            state: 状态
            action: 动作
            log_prob: 对数概率
            value: 状态价值
            reward: 奖励
            done: 是否结束
        """
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def update(self) -> float:
        """
        更新网络参数
        
        Returns:
            loss: 损失值
        """
        if len(self.states) < self.batch_size:
            return 0.0
        
        # 计算优势函数
        advantages = self._compute_advantages()
        
        # 转换为张量
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)
        old_log_probs = torch.stack(self.log_probs).detach()
        old_values = torch.stack(self.values).detach()
        
        # 多轮更新
        total_loss = 0.0
        for _ in range(self.epochs):
            # 随机打乱数据
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_old_values = old_values[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # 计算新概率和价值
                new_action_probs = self.actor(batch_states)
                dist = Categorical(new_action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                
                new_values = self.critic(batch_states).squeeze()
                
                # 计算比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # 计算actor损失
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 计算critic损失
                critic_loss = nn.MSELoss()(new_values, batch_old_values + batch_advantages)
                
                # 总损失
                loss = actor_loss + 0.5 * critic_loss
                
                # 反向传播
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                total_loss += loss.item()
        
        # 清空缓冲区
        self._clear_buffer()
        
        return total_loss / (self.epochs * (len(states) // self.batch_size))
    
    def _compute_advantages(self) -> torch.Tensor:
        """计算优势函数"""
        advantages = []
        returns = []
        
        # 计算回报
        R = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns)
        values = torch.stack(self.values)
        
        # 计算优势
        advantages = returns - values
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages
    
    def _clear_buffer(self):
        """清空经验缓冲区"""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
