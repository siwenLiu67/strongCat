import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, List

class SchedulingPolicy(nn.Module):
    """调度策略网络
    
    使用图注意力网络处理作业调度问题:
    1. 双重图结构: 作业图和机器图
    2. 交叉注意力: 处理作业-机器分配关系
    3. Actor-Critic结构: 输出调度策略和状态价值
    """
    def __init__(self, config):
        super().__init__()
        
         # 从配置中获取网络参数
        network_config = config.network.scheduling
        self.hidden_dim = network_config.gat_hidden_dim
        self.n_heads = network_config.gat_heads
        self.gat_layers = network_config.gat_layers
        self.node_dim = network_config.node_dim
        self.edge_dim = network_config.edge_dim
        self.action_dim = network_config.action_dim
        self.dropout = network_config.dropout
        self.activation = getattr(nn, network_config.activation.value.upper())()
        
        # 从配置获取特征维度
        self.job_feat_dim = self.node_dim      # 作业节点特征维度
        self.machine_feat_dim = self.node_dim   # 机器节点特征维度
        
        # 特征编码器
        self.encoders = nn.ModuleDict({
            'job': self._build_encoder(self.job_feat_dim),
            'machine': self._build_encoder(self.machine_feat_dim)
        })
        
        # 图注意力层
        self.gat_layers = nn.ModuleList([
            self._build_gat_layer() for _ in range(self.gat_layers)
        ])
        
        # 交叉注意力层
        self.cross_attention = CrossAttention(
            query_dim=self.hidden_dim,
            key_dim=self.hidden_dim,
            n_heads=self.n_heads,
            dropout=self.dropout
        )
        
        # Actor-Critic heads
        self.heads = nn.ModuleDict({
            'policy': self._build_mlp(self.hidden_dim * 2, self.action_dim),
            'value': self._build_mlp(self.hidden_dim * 2, 1)
        })
        
        # 从训练配置获取参数
        train_config = config.training.scheduling
        self.learning_rate = train_config.learning_rate
        self.gamma = train_config.gamma
        self.tau = train_config.tau
        self.n_step_returns = train_config.n_step_returns
        self.priority_alpha = train_config.priority_alpha
        self.priority_beta = train_config.priority_beta
        self.grad_clip = train_config.grad_clip
        self.value_loss_coef = train_config.value_loss_coef
        self.entropy_coef = train_config.entropy_coef
        
        # 初始化优化器
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)        
        
    def _build_encoder(self, input_dim: int) -> nn.Module:
        """构建特征编码器"""
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            self.activation,
            nn.Dropout(self.dropout)
        )
        
    def _build_gat_layer(self) -> nn.Module:
        """构建GAT层"""
        return GATLayer(
            in_dim=self.hidden_dim,
            out_dim=self.hidden_dim,
            n_heads=self.n_heads,
            dropout=self.dropout,
            activation=self.activation
        )
        
    def _build_mlp(self, input_dim: int, output_dim: int) -> nn.Module:
        """构建多层感知机"""
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            self.activation,
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, output_dim)
        )

    def act(self, job_features, job_adj, machine_features, machine_adj, 
            deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """选择调度动作
        
        Args:
            job_features: 作业特征
            job_adj: 作业邻接矩阵
            machine_features: 机器特征
            machine_adj: 机器邻接矩阵
            deterministic: 是否使用确定性策略
            
        Returns:
            action: 选择的调度动作
            action_log_prob: 动作对数概率
        """
        probs, _ = self.forward(job_features, job_adj, machine_features, machine_adj)
        
        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            action = torch.multinomial(probs, 1)
        
        action_log_prob = torch.log(torch.gather(probs, -1, action))
        
        return action, action_log_prob

    def evaluate_actions(self, job_features, job_adj, machine_features, machine_adj,
                        actions) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """评估动作的价值
        
        Returns:
            action_log_probs: 动作对数概率
            state_values: 状态价值
            entropy: 策略熵
        """
        probs, state_values = self.forward(job_features, job_adj, machine_features, machine_adj)
        
        action_log_probs = torch.log(torch.gather(probs, -1, actions))
        entropy = -(probs * torch.log(probs)).sum(-1)
        
        return action_log_probs, state_values, entropy

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """更新策略网络
        
        Args:
            batch: 训练数据批次
            
        Returns:
            metrics: 训练指标字典
        """
        # 计算n步回报
        returns = self._compute_returns(
            batch['rewards'],
            batch['dones'],
            batch['next_values'],
            self.gamma,
            self.n_step_returns
        )
        
        # 评估动作
        log_probs, values, entropy = self.evaluate_actions(
            batch['job_features'],
            batch['job_adj'],
            batch['machine_features'],
            batch['machine_adj'],
            batch['actions']
        )
        
        # 计算损失
        advantages = returns - values.detach()
        policy_loss = -(log_probs * advantages).mean()
        value_loss = torch.nn.functional.mse_loss(values, returns)
        entropy_loss = -entropy.mean()
        
        # 计算损失时使用配置的系数
        loss = (policy_loss + 
                self.value_loss_coef * value_loss + 
                self.entropy_coef * entropy_loss)
        
        # 使用配置的梯度裁剪值
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.mean().item()
        }

    @staticmethod
    def _compute_returns(rewards: torch.Tensor,
                        dones: torch.Tensor,
                        next_values: torch.Tensor,
                        gamma: float,
                        n_steps: int) -> torch.Tensor:
        """计算n步回报"""
        returns = torch.zeros_like(rewards)
        future_return = next_values
        
        for t in reversed(range(rewards.size(0))):
            returns[t] = rewards[t] + gamma * future_return * (1.0 - dones[t])
            future_return = returns[t]
            
        return returns

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, List, Optional

class CrossAttention(nn.Module):
    """交叉注意力模块
    
    用于处理作业和机器之间的注意力机制：
    1. 作业特征作为查询(query)
    2. 机器特征作为键(key)和值(value)
    3. 使用多头注意力机制提取作业-机器间的关联
    """
    def __init__(self, 
                 query_dim: int,      # 查询向量维度(作业特征)
                 key_dim: int,        # 键值向量维度(机器特征)
                 n_heads: int = 4,    # 注意力头数
                 dropout: float = 0.1):
        super().__init__()
        
        self.n_heads = n_heads
        self.head_dim = query_dim // n_heads
        self.scale = self.head_dim ** -0.5  # 缩放因子
        
        # 多头线性变换层
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(key_dim, query_dim)
        self.v_proj = nn.Linear(key_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)
        
        # Layer Norm和Dropout
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 前馈网络(FFN)
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_dim * 4, query_dim)
        )
        
    def forward(self, 
                query: torch.Tensor,     # [batch_size, n_jobs, query_dim]
                key: torch.Tensor,       # [batch_size, n_machines, key_dim]
                mask: Optional[torch.Tensor] = None  # [batch_size, n_jobs, n_machines]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播
        
        Args:
            query: 查询张量,形状为 [batch_size, n_jobs, query_dim]
            key: 键值张量,形状为 [batch_size, n_machines, key_dim]
            mask: 可选的注意力掩码,形状为 [batch_size, n_jobs, n_machines]
            
        Returns:
            output: 注意力输出,形状为 [batch_size, n_jobs, query_dim]
            attn_weights: 注意力权重,形状为 [batch_size, n_heads, n_jobs, n_machines]
        """
        batch_size = query.size(0)
        n_jobs = query.size(1)
        n_machines = key.size(1)
        
        # 1. 计算Q、K、V
        q = self.q_proj(query)  # [batch_size, n_jobs, query_dim]
        k = self.k_proj(key)    # [batch_size, n_machines, query_dim]
        v = self.v_proj(key)    # [batch_size, n_machines, query_dim]
        
        # 2. 分离多头
        q = q.view(batch_size, n_jobs, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, n_machines, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, n_machines, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 3. 缩放点积注意力
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 4. 应用mask(如果有)
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        
        # 5. Softmax获取注意力权重
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 6. 注意力加权求和
        out = torch.matmul(attn_weights, v)  # [batch_size, n_heads, n_jobs, head_dim]
        
        # 7. 合并多头
        out = out.transpose(1, 2).contiguous().view(batch_size, n_jobs, -1)
        
        # 8. 输出投影
        out = self.out_proj(out)
        
        # 9. 残差连接和层标准化
        query = query + self.dropout(out)
        query = self.norm1(query)
        
        # 10. 前馈网络
        ff_out = self.ffn(query)
        out = query + self.dropout(ff_out)
        out = self.norm2(out)
        
        return out, attn_weights

    def extra_repr(self) -> str:
        """返回额外的字符串表示"""
        return f'n_heads={self.n_heads}, head_dim={self.head_dim}'
    


class GATLayer(nn.Module):
    """图注意力层
    
    实现图注意力网络(GAT)层,用于:
    1. 提取节点间的关系
    2. 聚合邻居节点信息
    3. 更新节点表示
    """
    def __init__(self,
                 in_dim: int,         # 输入特征维度
                 out_dim: int,        # 输出特征维度
                 n_heads: int = 4,    # 注意力头数
                 dropout: float = 0.1, # dropout比率
                 activation: nn.Module = nn.GELU(), # 激活函数
                 residual: bool = True, # 是否使用残差连接
                 layer_norm: bool = True): # 是否使用层标准化
        super().__init__()
        
        self.n_heads = n_heads
        self.activation = activation
        self.residual = residual
        self.head_dim = out_dim // n_heads
        
        # 特征转换
        self.fc = nn.Linear(in_dim, out_dim)
        
        # 注意力机制
        self.attn = nn.Parameter(torch.empty(1, n_heads, self.head_dim * 2))
        nn.init.xavier_uniform_(self.attn)
        
        # Layer Norm和Dropout
        self.norm = nn.LayerNorm(out_dim) if layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        
        # 输出投影
        self.out_proj = nn.Linear(out_dim, out_dim)
        
        # 如果输入输出维度不同,创建残差映射
        self.res_proj = None
        if residual and in_dim != out_dim:
            self.res_proj = nn.Linear(in_dim, out_dim)
            
    def forward(self, 
                x: torch.Tensor,              # 节点特征 [batch_size, n_nodes, in_dim]
                adj: torch.Tensor,            # 邻接矩阵 [batch_size, n_nodes, n_nodes]
                mask: Optional[torch.Tensor] = None  # 节点掩码 [batch_size, n_nodes]
                ) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入节点特征
            adj: 邻接矩阵(0-1矩阵)
            mask: 可选的节点掩码
            
        Returns:
            out: 更新后的节点特征 [batch_size, n_nodes, out_dim]
        """
        batch_size, n_nodes, _ = x.shape
        
        # 1. 特征转换
        h = self.fc(x)  # [batch_size, n_nodes, out_dim]
        
        # 2. 分离多头
        h = h.view(batch_size, n_nodes, self.n_heads, self.head_dim)
        
        # 3. 计算注意力分数
        # 自身特征
        self_h = h.unsqueeze(2).expand(-1, -1, n_nodes, -1, -1)
        # 邻居特征
        neigh_h = h.unsqueeze(1).expand(-1, n_nodes, -1, -1, -1)
        # 拼接特征
        pair_h = torch.cat([self_h, neigh_h], dim=-1)
        
        # 计算注意力权重
        attn = (pair_h * self.attn.unsqueeze(1).unsqueeze(1)).sum(dim=-1)
        
        # 4. 应用邻接矩阵掩码
        attn = attn.masked_fill(adj.unsqueeze(-1) == 0, float('-inf'))
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(-1)  # [batch_size, 1, n_nodes, 1]
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        # 5. Softmax获取注意力权重
        attn = torch.softmax(attn, dim=2)  # 在邻居维度上做softmax
        attn = self.dropout(attn)
        
        # 6. 聚合邻居信息
        # [batch_size, n_nodes, n_nodes, n_heads, 1] * [batch_size, n_nodes, n_nodes, n_heads, head_dim]
        out = (attn.unsqueeze(-1) * neigh_h).sum(dim=2)
        
        # 7. 合并多头
        out = out.reshape(batch_size, n_nodes, -1)
        
        # 8. 输出投影
        out = self.out_proj(out)
        
        # 9. 残差连接
        if self.residual:
            if self.res_proj is not None:
                x = self.res_proj(x)
            out = x + self.dropout(out)
            
        # 10. 层标准化和激活
        out = self.norm(out)
        out = self.activation(out)
        
        return out

    def extra_repr(self) -> str:
        """返回额外的字符串表示"""
        return f'n_heads={self.n_heads}, head_dim={self.head_dim}'