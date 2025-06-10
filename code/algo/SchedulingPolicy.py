import torch
import torch.nn as nn
from algo.hdrl_framework import GATLayer, CrossAttention

class SchedulingPolicy(nn.Module):
    """调度策略网络
    
    使用图注意力网络处理作业调度问题:
    1. 双重图结构: 作业图和机器图
    2. 交叉注意力: 处理作业-机器分配关系
    3. Actor-Critic结构: 输出调度策略和状态价值
    """
    def __init__(self, config):
        super().__init__()
        
        # 配置参数初始化
        self.hidden_dim = config.get('hidden_dim', 64)    # 隐藏层维度
        self.job_feat_dim = 7       # 作业特征维度: 4(基础) + 3(调度)
        self.machine_feat_dim = 4   # 机器特征维度
        self.n_heads = config.get('n_heads', 4)  # 注意力头数
        self.dropout = config.get('dropout', 0.1) # dropout率
        
        # 特征编码器 - 将原始特征映射到隐藏空间
        self.encoders = nn.ModuleDict({
            'job': self._build_encoder(self.job_feat_dim),
            'machine': self._build_encoder(self.machine_feat_dim)
        })
        
        # 图注意力层 - 分别处理作业图和机器图
        self.gat_layers = nn.ModuleDict({
            'job': self._build_gat_layers(),
            'machine': self._build_gat_layers()
        })
        
        # 交叉注意力层 - 处理作业-机器匹配关系
        self.cross_attention = CrossAttention(
            query_dim=self.hidden_dim,
            key_dim=self.hidden_dim,
            n_heads=self.n_heads
        )
        
        # 策略头和价值头 - Actor-Critic结构
        self.heads = nn.ModuleDict({
            'policy': self._build_mlp(self.hidden_dim * 2, 1),  # 输出分配概率
            'value': self._build_mlp(self.hidden_dim * 2, 1)   # 输出状态价值
        })
        
        self._init_weights()  # 初始化网络权重
        
    def _build_encoder(self, input_dim):
        """构建特征编码器"""
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
    
    def _build_gat_layers(self):
        """构建图注意力层"""
        return nn.ModuleList([
            GATLayer(self.hidden_dim, self.hidden_dim) for _ in range(2)
        ])
    
    def _build_mlp(self, input_dim, output_dim):
        """构建多层感知机"""
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, output_dim)
        )
        
    def forward(self, job_features, job_adj, machine_features, machine_adj):
        """前向传播
        
        Args:
            job_features: [batch_size, n_jobs, job_feat_dim] 作业特征
            job_adj: [batch_size, n_jobs, n_jobs] 作业邻接矩阵
            machine_features: [batch_size, n_machines, machine_feat_dim] 机器特征
            machine_adj: [batch_size, n_machines, n_machines] 机器邻接矩阵
            
        Returns:
            schedule_probs: [batch_size, n_jobs, n_machines] 调度概率矩阵
            state_value: [batch_size, 1] 状态价值
        """
        # 1. 特征编码
        job_embed = self.encoders['job'](job_features)
        machine_embed = self.encoders['machine'](machine_features)
        
        # 2. 图特征提取
        for gat in self.gat_layers['job']:
            job_embed = gat(job_embed, job_adj)
        for gat in self.gat_layers['machine']:
            machine_embed = gat(machine_embed, machine_adj)
            
        # 3. 作业-机器匹配特征
        assignment_features = self.cross_attention(job_embed, machine_embed)
        
        # 4. 生成分配概率矩阵
        batch_size, n_jobs, _ = job_features.shape
        n_machines = machine_features.size(1)
        
        # 计算所有可能的作业-机器组合
        job_idx = torch.arange(n_jobs).repeat_interleave(n_machines)
        machine_idx = torch.arange(n_machines).repeat(n_jobs)
        
        # 合并作业和机器特征
        combined_features = torch.cat([
            assignment_features[:, job_idx],
            machine_embed[:, machine_idx]
        ], dim=-1)
        
        # 5. 输出调度策略和状态价值
        logits = self.heads['policy'](combined_features)
        probs = torch.softmax(
            logits.view(batch_size, n_jobs, n_machines), 
            dim=-1
        )
        
        # 6. 计算状态价值
        global_features = torch.cat([
            job_embed.mean(dim=1),
            machine_embed.mean(dim=1)
        ], dim=-1)
        value = self.heads['value'](global_features)
        
        return probs, value