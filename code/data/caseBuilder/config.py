from dataclasses import dataclass, field
from typing import Dict, Any, Literal
from enum import Enum
import json
from pathlib import Path

class ActivationType(str, Enum):
    """激活函数类型"""
    RELU = 'relu'
    GELU = 'gelu'
    TANH = 'tanh'

@dataclass
class NetworkConfig:
    """网络架构配置"""
    # 状态空间
    state_dim: int = 128        # 元控制器状态向量维度
    node_dim: int = 64         # 图节点特征维度
    edge_dim: int = 32         # 图边特征维度
    
    # 动作空间
    action_dim: int = 10       # 元控制器动作维度
    wh_action_dim: int = 20    # 仓储动作维度
    dist_action_dim: int = 15  # 配送动作维度
    
    # 网络结构
    transformer_layers: int = 3  # Transformer层数
    transformer_heads: int = 4   # 注意力头数
    gat_hidden_dim: int = 64    # GAT隐藏层维度
    dropout: float = 0.1        # Dropout比率
    activation: ActivationType = ActivationType.GELU

@dataclass
class TrainingConfig:
    """训练相关配置"""
    # 基础参数
    gamma: float = 0.99         # 折扣因子
    tau: float = 0.005         # 目标网络软更新系数
    batch_size: int = 64       # 训练批次大小
    buffer_size: int = 100000  # 经验回放缓冲区大小
    learning_rate: float = 1e-4 # 学习率
    grad_clip: float = 0.5     # 梯度裁剪阈值
    
    # 评估参数
    eval_episodes: int = 50     # 评估轮数
    save_interval: int = 100    # 模型保存间隔

@dataclass
class ProblemConfig:
    """问题规模和约束配置"""
    # 规模参数
    num_jobs: int = 20         # 工件数量
    num_machines: int = 10     # 机器数量
    num_distributors: int = 5  # 配送商数量
    
    # 工序参数
    min_operations: int = 3    # 最少工序数
    max_operations: int = 6    # 最多工序数
    min_machines_per_op: int = 1  # 每道工序最少可选机器数
    max_machines_per_op: int = 4  # 每道工序最多可选机器数
    min_processing_time: int = 3  # 最短加工时间
    max_processing_time: int = 10 # 最长加工时间
    avg_processing_time: int = 5  # 平均加工时间
    
    # 配送参数
    urgent_threshold: int = 5   # 紧急配送阈值(分钟)
    min_delivery_requirements: int = 1  # 最少交付要求数
    max_delivery_requirements: int = 3  # 最多交付要求数
    earliest_delivery_time: int = 21 * 60  # 最早交付时间(21:00)
    latest_delivery_time: int = 26 * 60    # 最晚交付时间(次日2:00)
    
    # 批次参数
    batch_loading_base_time: int = 10  # 批次基础装载时间
    item_loading_time: int = 2        # 单件装载时间
    max_jobs_per_batch: int = 5       # 每个批次最大作业数
    min_load_ratio: float = 0.3       # 最小装载率

@dataclass
class RewardConfig:
    """奖励权重配置"""
    utilization_weight: float = 0.8   # 机器利用率权重
    loading_weight: float = 0.5       # 装载效率权重
    violation_penalty: float = -1.0    # 约束违反惩罚
    scheduling_weight: float = 0.2    # 调度效率权重
    batching_weight: float = 0.3      # 配送效率权重
    tardiness_weight: float = 1.0     # 延迟惩罚权重

@dataclass
class Config:
    """FJSP-DP问题主配置类"""
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    problem: ProblemConfig = field(default_factory=ProblemConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    
    random_seed: int = 42  # 随机种子
    max_time_steps: int = 10000  # 最大时间步数
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典格式"""
        return {
            'network': self.network.__dict__,
            'training': self.training.__dict__,
            'problem': self.problem.__dict__,
            'reward': self.reward.__dict__,
            'random_seed': self.random_seed,
            'max_time_steps': self.max_time_steps
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """从字典创建配置对象"""
        network = NetworkConfig(**config_dict.get('network', {}))
        training = TrainingConfig(**config_dict.get('training', {}))
        problem = ProblemConfig(**config_dict.get('problem', {}))
        reward = RewardConfig(**config_dict.get('reward', {}))
        
        return cls(
            network=network,
            training=training,
            problem=problem,
            reward=reward,
            random_seed=config_dict.get('random_seed', 42),
            max_time_steps=config_dict.get('max_time_steps', 10000)
        )
    
    def save(self, filepath: str | Path) -> None:
        """保存配置到文件"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str | Path) -> 'Config':
        """从文件加载配置"""
        with open(filepath) as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)