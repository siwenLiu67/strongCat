import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Dict, List, Tuple, Optional


class FeatureExtractor:
    """特征提取器基类"""
    def __init__(self, config: Dict):
        self.config = config
        
    def normalize_time(self, time_value: float) -> float:
        """时间归一化"""
        return time_value / self.config['max_time']
    
    def normalize_weight(self, weight: float) -> float:
        """重量归一化"""
        return weight / self.config['max_weight']

class JobFeatures(FeatureExtractor):
    """作业特征提取器"""
    def get_basic_features(self, job: Dict) -> List[float]:
        """获取基础特征"""
        return [
            self.normalize_time(job['due_date']),         # 截止日期
            job['priority'] / self.config['max_priority'], # 优先级
            self.normalize_weight(job['weight']),         # 重量
            len(job.get('operations', [])) / self.config['max_operations'] # 工序数
        ]
    
    def get_scheduling_features(self, job: Dict) -> List[float]:
        """获取调度相关特征"""
        basic_features = self.get_basic_features(job)
        scheduling_features = [
            len(job['remaining_operations']) / len(job['operations']), # 剩余工序比例
            self.normalize_time(job['processing_time']),              # 加工时间
            job['machine_compatibility'] / self.config['num_machines'] # 机器兼容性
        ]
        return basic_features + scheduling_features
    
    def get_dispatching_features(self, job: Dict, current_time: float) -> List[float]:
        """获取配送相关特征"""
        basic_features = self.get_basic_features(job)
        dispatching_features = [
            self.normalize_time(job['completion_time'] - current_time),  # 完工时间
            job['distributor_id'] / self.config['num_distributors'],     # 配送商ID
            self.normalize_time(job.get('tardiness', 0))                # 延迟时间
        ]
        return basic_features + dispatching_features

class MachineFeatures(FeatureExtractor):
    """机器特征提取器"""
    def get_features(self, machine: Dict) -> List[float]:
        return [
            len(machine['queue']) / self.config['max_queue_length'],   # 队列长度
            machine['utilization'],                                     # 利用率
            self.normalize_time(machine['remaining_time']),            # 剩余时间
            machine['status'] == 'idle'                                # 是否空闲
        ]

class BatchFeatures(FeatureExtractor):
    """批次特征提取器"""
    def get_features(self, batch: Dict, current_time: float) -> List[float]:
        current_load = sum(j['weight'] for j in batch['assigned_jobs'])
        return [
            (batch['max_capacity'] - current_load) / batch['max_capacity'], # 剩余容量比例
            self.normalize_time(batch['earliest_start'] - current_time),   # 最早开始时间
            self.normalize_time(batch['latest_start'] - current_time),     # 最晚开始时间
            len(batch['assigned_jobs']) / self.config['max_batch_size'],   # 已分配数量
            batch['distributor_id'] / self.config['num_distributors'],     # 配送商ID
            batch.get('utilization', 0)                                    # 当前利用率
        ]

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
    
    def __init__(self, 
                 state_dim=5,  # 5个全局指标
                 action_dim=3, # 高层动作数量
                 hidden_dim=64,
                 dropout=0.1):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 特征提取层
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Dueling网络结构
        # 价值流
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 优势流
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'norm' in name.lower():
                    # LayerNorm层的权重初始化为1
                    nn.init.constant_(param.data, 1.0)
                else:
                    # 线性层使用正交初始化
                    nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                # 所有偏置初始化为0
                nn.init.constant_(param.data, 0.0)
                
    def forward(self, x):
        """前向传播
        
        参数:
            x (Tensor): 输入状态张量, shape [batch_size, state_dim]
            
        返回:
            Tensor: Q值, shape [batch_size, action_dim]
        """
        # 特征提取
        features = self.feature_layer(x)
        
        # Dueling架构
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Q = V + (A - mean(A))
        q_value = value + advantage - advantage.mean(dim=-1, keepdim=True)
        
        return q_value
    
    def predict(self, state, temperature=1.0):
        """带温度参数的动作选择
        
        参数:
            state (Tensor): 输入状态
            temperature (float): 温度参数，控制探索程度
            
        返回:
            Tensor: 选择的动作
        """
        q_values = self.forward(state)
        
        # 应用温度缩放
        scaled_q = q_values / temperature
        
        # 使用softmax获取动作概率
        action_probs = torch.softmax(scaled_q, dim=-1)
        
        # 按概率采样动作
        action = torch.multinomial(action_probs, 1)
        
        return action
    


class CrossAttention(nn.Module):
    """交叉注意力层：用于作业-机器匹配"""
    def __init__(self, query_dim, key_dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = query_dim // n_heads
        
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(key_dim, query_dim)
        self.v_proj = nn.Linear(key_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)
        
    def forward(self, queries, keys):
        B, Nq, _ = queries.shape
        _, Nk, _ = keys.shape
        
        # 多头投影
        q = self.q_proj(queries).view(B, Nq, self.n_heads, self.head_dim)
        k = self.k_proj(keys).view(B, Nk, self.n_heads, self.head_dim)
        v = self.v_proj(keys).view(B, Nk, self.n_heads, self.head_dim)
        
        # 计算注意力分数
        scores = torch.einsum('bqhd,bkhd->bhqk', q, k)
        scores = scores / np.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        
        # 聚合结果
        out = torch.einsum('bhqk,bkhd->bqhd', attn, v)
        out = out.reshape(B, Nq, -1)
        
        return self.out_proj(out)

class GATLayer(nn.Module):
    """Graph Attention Layer"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Linear(in_features, out_features)
        self.a = nn.Linear(2*out_features, 1)
        
    def forward(self, h, adj):
        h = self.W(h)
        N = h.size(0)
        
        # Attention mechanism
        a_input = torch.cat([h.repeat(1,N).view(N*N, -1), 
                           h.repeat(N,1)], dim=1)
        e = self.a(a_input).squeeze(1)
        e = e.view(N, N)
        e = e.masked_fill(adj == 0, -1e9)
        attention = torch.softmax(e, dim=1)
        
        return torch.matmul(attention, h)


class FeatureEncoder(nn.Module):
    """特征编码器"""
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DispatchingAttention(nn.Module):
    """配送注意力层"""
    def __init__(self, hidden_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.job_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout)
        self.batch_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout)
        
    def forward(self, 
                job_embed: torch.Tensor,
                batch_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 自注意力
        job_embed = self.job_attn(job_embed, job_embed, job_embed)[0]
        batch_embed = self.batch_attn(batch_embed, batch_embed, batch_embed)[0]
        
        # 交叉注意力
        job_embed = self.cross_attn(job_embed, batch_embed, batch_embed)[0]
        
        return job_embed, batch_embed

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
    
def create_dispatching_features(
    jobs: List[Dict],
    batches: List[Dict],
    current_time: float,
    config: Dict
) -> Dict[str, torch.Tensor]:
    """创建配送决策的输入特征
    
    Args:
        jobs: 工件列表
        batches: 批次列表
        current_time: 当前时间
        config: 配置参数
    
    Returns:
        包含特征张量的字典
    """
    # 工件特征
    job_features = []
    distributor_ids = []
    
    for job in jobs:
        features = [
            (job['completion_time'] - current_time) / config['max_time'],  # 归一化完工时间
            (job['due_date'] - current_time) / config['max_time'],        # 归一化截止期
            job['priority'] / config['max_priority'],                      # 归一化优先级
            job['weight'] / config['max_weight'],                         # 归一化重量
            len(job.get('operations', [])) / config['max_operations'],    # 归一化工序数
            job.get('tardiness', 0) / config['max_time']                 # 归一化延迟
        ]
        job_features.append(features)
        distributor_ids.append(job['distributor_id'])
    
    # 批次特征
    batch_features = []
    for batch in batches:
        current_load = sum(j['weight'] for j in batch['assigned_jobs'])
        remaining_capacity = batch['max_capacity'] - current_load
        
        features = [
            remaining_capacity / batch['max_capacity'],                    # 归一化剩余容量
            (batch['earliest_start'] - current_time) / config['max_time'], # 归一化最早开始时间
            (batch['latest_start'] - current_time) / config['max_time'],   # 归一化最晚开始时间
            len(batch['assigned_jobs']) / config['max_batch_size'],        # 归一化已分配数量
            batch['distributor_id'] / config['num_distributors'],          # 归一化配送商ID
            batch.get('utilization', 0)                                    # 当前利用率
        ]
        batch_features.append(features)
    
    return {
        'job_features': torch.tensor(job_features, dtype=torch.float32),
        'batch_features': torch.tensor(batch_features, dtype=torch.float32),
        'distributor_ids': torch.tensor(distributor_ids, dtype=torch.long)
    }

def create_valid_mask(
    jobs: List[Dict],
    batches: List[Dict],
    current_time: float,
    config: Dict
) -> torch.Tensor:
    """创建有效分配掩码
    
    考虑所有约束条件：
    1. 配送商匹配
    2. 容量限制
    3. 时间窗口约束
    """
    n_jobs = len(jobs)
    n_batches = len(batches)
    mask = torch.zeros((n_jobs, n_batches), dtype=torch.bool)
    
    for j, job in enumerate(jobs):
        for b, batch in enumerate(batches):
            # 检查所有约束
            if (job['distributor_id'] == batch['distributor_id'] and
                _check_capacity_constraint(job, batch, config) and
                _check_time_window_constraint(job, batch, current_time, config)):
                mask[j, b] = True
    
    return mask

def _check_capacity_constraint(job: Dict, batch: Dict, config: Dict) -> bool:
    """检查容量约束"""
    current_load = sum(j['weight'] for j in batch['assigned_jobs'])
    return current_load + job['weight'] <= batch['max_capacity']

def _check_time_window_constraint(
    job: Dict,
    batch: Dict,
    current_time: float,
    config: Dict
) -> bool:
    """检查时间窗口约束"""
    n_jobs = len(batch['assigned_jobs']) + 1
    processing_time = config['base_processing_time'] + n_jobs * config['per_job_time']
    completion_time = current_time + processing_time
    return completion_time <= batch['latest_start']