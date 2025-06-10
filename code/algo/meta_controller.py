import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Dict, List, Tuple, Optional
from torch.nn import functional as F

class MetaController(nn.Module):
    """元控制器实现
    
    使用Transformer架构的高层决策网络，结合Dueling DQN结构来分离状态价值和优势函数。
    输入状态包含5个全局指标：
    1. q_t: 输入缓冲区中的平均作业数量
    2. sigma_mach_t: 各机器剩余加工时间的标准差(工作负载不平衡指标)
    3. s_due_t: 缓冲区作业的平均时间裕度
    4. rho_urg_t: 缓冲区中紧急作业的比例
    5. mu_load_t: 系统中所有作业按订单类型的估计加工负载
    """
    
    def __init__(self, config):
        super().__init__()
        
        # 从配置中获取网络参数
        network_config = config.network.meta_controller
        self.state_dim = network_config.state_dim
        self.action_dim = network_config.action_dim
        self.hidden_dim = network_config.hidden_dim
        self.dropout = network_config.dropout
        self.activation = getattr(nn, network_config.activation.value.upper())()
        
        # 特征提取层
        self.feature_layer = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            self.activation,
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            self.activation,
            nn.Dropout(self.dropout)
        )
        
        # Dueling网络结构
        # 价值流
        self.value_stream = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            self.activation,
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
        # 优势流
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            self.activation,
            nn.Linear(self.hidden_dim // 2, self.action_dim)
        )
        
        # 训练相关参数
        train_config = config.training.meta_controller
        self.learning_rate = train_config.learning_rate
        self.gamma = train_config.gamma
        self.tau = train_config.tau
        self.batch_size = train_config.batch_size
        self.update_freq = train_config.update_freq
        self.entropy_coef = train_config.entropy_coef
        self.clip_param = train_config.clip_param
        
        # 初始化优化器
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        

    def act(self, state: torch.Tensor, epsilon: float = 0.1) -> Tuple[torch.Tensor, float]:
        """选择动作
        
        Args:
            state: 状态向量 [batch_size, state_dim]
            epsilon: ε-贪婪探索的概率
            
        Returns:
            action: 选择的动作
            q_value: 对应的Q值
        """
        if random.random() < epsilon:
            # 随机探索
            action = torch.randint(0, self.action_dim, (state.size(0),))
            with torch.no_grad():
                q_value = self.forward(state).gather(1, action.unsqueeze(-1))
        else:
            # 贪婪选择
            with torch.no_grad():
                q_values = self.forward(state)
                action = q_values.argmax(dim=-1)
                q_value = q_values.gather(1, action.unsqueeze(-1))
        
        return action, q_value

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """更新元控制器网络
        
        使用Dueling DQN算法更新网络参数:
        1. 计算当前Q值
        2. 计算目标Q值
        3. 计算TD误差和损失
        4. 更新网络参数
        
        Args:
            batch: 训练数据批次，包含:
                - states: 状态向量 [batch_size, state_dim]
                - actions: 选择的动作 [batch_size]
                - rewards: 获得的奖励 [batch_size]
                - next_states: 下一状态 [batch_size, state_dim]
                - dones: 终止标志 [batch_size]
                
        Returns:
            训练信息字典，包含各类损失值
        """
        # 1. 从当前状态获取Q值
        current_features = self.feature_layer(batch['states'])
        current_value = self.value_stream(current_features)
        current_advantage = self.advantage_stream(current_features)
        current_q = current_value + (
            current_advantage - 
            current_advantage.mean(dim=-1, keepdim=True)
        )
        
        # 获取所选动作的Q值
        current_q_selected = torch.gather(
            current_q, 1, 
            batch['actions'].unsqueeze(-1)
        )
        
        # 2. 计算目标Q值
        with torch.no_grad():
            next_features = self.feature_layer(batch['next_states'])
            next_value = self.value_stream(next_features)
            next_advantage = self.advantage_stream(next_features)
            next_q = next_value + (
                next_advantage - 
                next_advantage.mean(dim=-1, keepdim=True)
            )
            
            # Double DQN: 使用当前网络选择动作，目标网络评估动作
            next_actions = next_q.argmax(dim=-1, keepdim=True)
            next_q_selected = torch.gather(next_q, 1, next_actions)
            
            # 计算目标值
            target_q = batch['rewards'].unsqueeze(-1) + (
                (1 - batch['dones'].unsqueeze(-1)) * 
                self.gamma * next_q_selected
            )
        
        # 3. 计算各种损失
        # TD误差
        td_error = (target_q - current_q_selected).abs()
        
        # 值损失（使用Huber损失）
        value_loss = F.smooth_l1_loss(current_q_selected, target_q)
        
        # 优势损失
        advantage_loss = F.mse_loss(
            current_advantage.mean(), 
            torch.zeros_like(current_advantage.mean())
        )
        
        # 熵正则化
        probs = F.softmax(current_q, dim=-1)
        entropy_loss = -(probs * torch.log(probs + 1e-10)).sum(-1).mean()
        
        # 总损失
        loss = (
            value_loss + 
            0.5 * advantage_loss - 
            self.entropy_coef * entropy_loss
        )
        
        # 4. 更新网络
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_param)
        self.optimizer.step()
        
        return {
            'total_loss': loss.item(),
            'value_loss': value_loss.item(),
            'advantage_loss': advantage_loss.item(),
            'entropy': entropy_loss.item(),
            'td_error': td_error.mean().item(),
            'q_value': current_q_selected.mean().item()
        }




class PolicyNetwork(nn.Module):
    """策略网络"""
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, job_embed: torch.Tensor, batch_embed: torch.Tensor) -> torch.Tensor:
        B, N_j, _ = job_embed.shape
        _, N_b, _ = batch_embed.shape
        
        # 展开所有可能的工件-批次对
        job_expanded = job_embed.unsqueeze(2).expand(-1, -1, N_b, -1)
        batch_expanded = batch_embed.unsqueeze(1).expand(-1, N_j, -1, -1)
        
        # 合并特征并计算logits
        combined = torch.cat([job_expanded, batch_expanded], dim=-1)
        return self.net(combined).squeeze(-1)

class ValueNetwork(nn.Module):
    """价值网络"""
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, job_embed: torch.Tensor, batch_embed: torch.Tensor) -> torch.Tensor:
        # 全局池化
        global_job = job_embed.mean(dim=1)
        global_batch = batch_embed.mean(dim=1)
        
        # 合并特征并计算状态值
        global_feat = torch.cat([global_job, global_batch], dim=-1)
        return self.net(global_feat)
    
