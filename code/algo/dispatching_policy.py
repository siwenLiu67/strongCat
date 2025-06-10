from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  


class DispatchingPolicy(nn.Module):
    """配送策略网络
    
    使用分层注意力结构处理工件分配到批次的决策：
    1. 工件编码：提取工件特征(完工时间、截止期等)
    2. 批次编码：提取批次特征(容量、时间窗等)
    3. 分配决策：基于工件-批次匹配生成分配概率
    """
    def __init__(self, config):
        super().__init__()
        
        # 从配置中获取网络参数
        network_config = config.network.dispatching
        self.hidden_dim = network_config.hidden_dim
        self.lstm_layers = network_config.lstm_layers
        self.action_dim = network_config.action_dim
        self.dropout = network_config.dropout
        self.activation = getattr(nn, network_config.activation.value.upper())()
        
        # 特征维度设置
        self.job_feat_dim = 7    # 工件特征维度: 4(基础) + 3(配送)
        self.batch_feat_dim = 6   # 批次特征维度
        
        # 特征编码器
        self.encoders = nn.ModuleDict({
            'job': self._build_encoder(self.job_feat_dim),
            'batch': self._build_encoder(self.batch_feat_dim)
        })
        
        # 分层注意力层
        self.attention_layers = nn.ModuleList([
            DispatchingAttention(
                hidden_dim=self.hidden_dim,
                n_heads=network_config.transformer_heads,
                dropout=self.dropout,
                activation=self.activation
            ) for _ in range(network_config.transformer_layers)
        ])
        
        # Actor-Critic heads
        self.heads = nn.ModuleDict({
            'policy': self._build_mlp(self.hidden_dim * 2, self.action_dim),
            'value': self._build_mlp(self.hidden_dim * 2, 1)
        })
        
        # 从训练配置获取参数
        train_config = config.training.dispatching
        self.learning_rate = train_config.learning_rate
        self.gamma = train_config.gamma
        self.tau = train_config.tau
        self.batch_size = train_config.batch_size
        self.sac_alpha = train_config.sac_alpha
        self.auto_entropy_tuning = train_config.auto_entropy_tuning
        self.reward_scale = train_config.reward_scale
        
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
        
    def _build_mlp(self, input_dim: int, output_dim: int) -> nn.Module:
        """构建多层感知机"""
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            self.activation,
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, output_dim)
        )

    def forward(self, batch: Dict[str, torch.Tensor], 
                valid_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 1. 特征编码
        job_embed = self.encoders['job'](batch['job_features'])
        batch_embed = self.encoders['batch'](batch['batch_features'])
        
        # 2. 多层注意力处理
        for attn in self.attention_layers:
            job_embed, batch_embed = attn(job_embed, batch_embed)
            
        # 3. 计算策略和价值
        combined = torch.cat([job_embed, batch_embed], dim=-1)
        policy = self.heads['policy'](combined)
        value = self.heads['value'](combined)
        
        # 4. 应用分配约束
        if valid_mask is not None:
            policy = policy.masked_fill(~valid_mask, float('-inf'))
            
        # 5. Softmax获取概率
        probs = torch.softmax(policy, dim=-1)
        
        return probs, value
        
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """更新策略网络"""
        # 计算SAC损失
        policy_loss, value_loss, entropy = self._compute_sac_loss(batch)
        
        # 总损失
        loss = (policy_loss + 
                value_loss + 
                self.sac_alpha * entropy)
                
        # 更新网络
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }
        


    def _compute_sac_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算SAC算法的损失
        
        Args:
            batch: 训练数据批次，包含:
                - job_features: 工件特征
                - batch_features: 批次特征
                - actions: 选择的动作
                - rewards: 奖励值
                - next_job_features: 下一状态工件特征
                - next_batch_features: 下一状态批次特征
                - dones: 终止标志
                - valid_mask: 有效动作掩码
                
        Returns:
            policy_loss: 策略损失
            value_loss: 价值损失
            entropy: 策略熵
        """
        # 1. 从当前状态获取策略和价值
        probs, value = self.forward(
            {
                'job_features': batch['job_features'],
                'batch_features': batch['batch_features']
            },
            batch.get('valid_mask')
        )
        
        # 2. 从下一状态获取目标价值
        with torch.no_grad():
            next_probs, next_value = self.forward(
                {
                    'job_features': batch['next_job_features'],
                    'batch_features': batch['next_batch_features']
                },
                batch.get('next_valid_mask')
            )
            
            # 计算熵
            next_entropy = -(next_probs * torch.log(next_probs + 1e-10)).sum(-1)
            
            # 计算目标Q值
            target_q = (batch['rewards'] * self.reward_scale + 
                       (1 - batch['dones']) * self.gamma * 
                       (next_value + self.sac_alpha * next_entropy))
        
        # 3. 计算策略损失
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum(-1)
        
        # 从动作中获取实际选择的概率
        chosen_probs = torch.gather(probs, -1, batch['actions'])
        chosen_log_probs = torch.log(chosen_probs + 1e-10)
        
        advantage = target_q.detach() - value
        policy_loss = -(chosen_log_probs * advantage.detach()).mean()
        
        # 4. 计算价值损失
        value_loss = F.mse_loss(value, target_q.detach())
        
        # 5. 如果启用自动调整熵参数
        if self.auto_entropy_tuning:
            target_entropy = -torch.prod(torch.tensor(probs.shape[1:])).item()
            entropy_diff = entropy.detach().mean() - target_entropy
            self.sac_alpha = self.sac_alpha * torch.exp(0.05 * entropy_diff)
            self.sac_alpha = torch.clamp(self.sac_alpha, 0.01, 1.0)
        
        return policy_loss, value_loss, entropy.mean()


class DispatchingAttention(nn.Module):
    """配送注意力层"""
    def __init__(self,
                 hidden_dim: int,
                 n_heads: int = 4,
                 dropout: float = 0.1,
                 activation: nn.Module = nn.GELU()):
        super().__init__()
        
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        # 多头注意力
        self.job_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.batch_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        
        # Layer Norm
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
    def forward(self, 
                job_embed: torch.Tensor,
                batch_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 1. 工件自注意力
        job_attn_out = self.job_attn(
            job_embed, job_embed, job_embed
        )[0]
        job_embed = self.norm1(job_embed + job_attn_out)
        
        # 2. 批次自注意力
        batch_attn_out = self.batch_attn(
            batch_embed, batch_embed, batch_embed
        )[0]
        batch_embed = self.norm1(batch_embed + batch_attn_out)
        
        # 3. FFN
        job_ffn_out = self.ffn(job_embed)
        batch_ffn_out = self.ffn(batch_embed)
        
        job_embed = self.norm2(job_embed + job_ffn_out)
        batch_embed = self.norm2(batch_embed + batch_ffn_out)
        
        return job_embed, batch_embed
