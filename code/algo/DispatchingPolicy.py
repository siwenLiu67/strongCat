from typing import Dict, List, Tuple, Optional
from torch import nn
import torch
from algo.hdrl_framework import FeatureEncoder
from algo.hdrl_framework import DispatchingAttention
from algo.hdrl_framework import PolicyNetwork
from algo.hdrl_framework import ValueNetwork


class DispatchingPolicy(nn.Module):
    """配送策略网络
    
    使用分层注意力结构处理工件分配到批次的决策：
    1. 工件编码：提取工件特征(完工时间、截止期等)
    2. 批次编码：提取批次特征(容量、时间窗等)
    3. 分配决策：基于工件-批次匹配生成分配概率
    """
    def __init__(self, 
                 job_feat_dim: int,
                 batch_feat_dim: int,
                 hidden_dim: int = 64,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        # 特征编码器
        self.encoders = nn.ModuleDict({
            'job': FeatureEncoder(
                input_dim=job_feat_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            ),
            'batch': FeatureEncoder(
                input_dim=batch_feat_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        })
        
        # 注意力层
        self.attention_layers = nn.ModuleList([
            DispatchingAttention(
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        
        # 策略网络
        self.policy_net = PolicyNetwork(
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # 价值网络
        self.value_net = ValueNetwork(
            hidden_dim=hidden_dim,
            dropout=dropout
        )
    
    def forward(self, 
                batch: Dict[str, torch.Tensor],
                valid_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播
        
        Args:
            batch: 包含以下键的字典:
                - job_features: [B, N_j, D_j] 工件特征
                - batch_features: [B, N_b, D_b] 批次特征
                - distributor_ids: [B, N_j] 工件所属配送商
            valid_mask: [B, N_j, N_b] 有效分配掩码
            
        Returns:
            dispatch_probs: [B, N_j, N_b] 分配概率
            state_value: [B, 1] 状态价值
        """
        # 1. 特征编码
        job_embed = self.encoders['job'](batch['job_features'])
        batch_embed = self.encoders['batch'](batch['batch_features'])
        
        # 2. 多层注意力处理
        for attn in self.attention_layers:
            job_embed, batch_embed = attn(job_embed, batch_embed)
        
        # 3. 生成分配概率
        logits = self.policy_net(job_embed, batch_embed)
        
        # 4. 应用分配约束
        if valid_mask is not None:
            logits = logits.masked_fill(~valid_mask, float('-inf'))
        
        # 5. 按配送商分组的softmax
        probs = self._distributor_grouped_softmax(
            logits, 
            batch['distributor_ids'],
            batch['batch_features'][..., -2]  # 假设批次特征中倒数第二维是distributor_id
        )
        
        # 6. 计算状态值
        value = self.value_net(job_embed, batch_embed)
        
        return probs, value
    
    @staticmethod
    def _distributor_grouped_softmax(logits: torch.Tensor,
                                   job_dist_ids: torch.Tensor,
                                   batch_dist_ids: torch.Tensor) -> torch.Tensor:
        """按配送商分组执行softmax
        
        确保工件只能分配给同一配送商的批次
        """
        B, N_j, N_b = logits.size()
        probs = torch.zeros_like(logits)
        
        for dist_id in torch.unique(job_dist_ids):
            # 创建当前配送商的掩码
            job_mask = (job_dist_ids == dist_id).unsqueeze(-1)
            batch_mask = (batch_dist_ids == dist_id).unsqueeze(1)
            dist_mask = job_mask & batch_mask
            
            # 对当前配送商的logits执行softmax
            dist_logits = logits.masked_fill(~dist_mask, float('-inf'))
            dist_probs = torch.softmax(dist_logits, dim=-1)
            probs = probs.masked_scatter(dist_mask, dist_probs[dist_mask])
            
        return probs
    
