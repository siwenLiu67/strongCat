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

# === 网络配置相关类 ===
@dataclass
class MetaControllerNetConfig:
    """元控制器网络配置"""
    state_dim: int = 128        # 状态向量维度
    action_dim: int = 10        # 动作维度
    hidden_dim: int = 256       # 隐藏层维度
    transformer_layers: int = 3  # Transformer层数
    transformer_heads: int = 4   # 注意力头数
    dropout: float = 0.1        # Dropout比率
    activation: ActivationType = ActivationType.GELU
    

    """元控制器配置"""
    max_jobs: int = 100            # 最大作业数量
    max_time: float = 1000.0       # 最大时间单位
    max_priority: int = 10         # 最大优先级
    max_load: float = 5000.0       # 最大加工负载
    urgent_time_threshold: float = 50.0  # 紧急作业时间阈值

    lr: float = 1e-4              # 学习率
    gamma: float = 0.99           # 折扣因子
    tau: float = 0.005            # 目标网络软更新系数
    


@dataclass
class SchedulingNetConfig:
    """调度策略网络配置"""
    node_dim: int = 64           # 图节点特征维度
    edge_dim: int = 32          # 图边特征维度
    action_dim: int = 20        # 调度动作维度
    gat_hidden_dim: int = 64    # GAT隐藏层维度
    gat_layers: int = 2         # GAT层数
    gat_heads: int = 4          # GAT注意力头数
    dropout: float = 0.1        # Dropout比率
    activation: ActivationType = ActivationType.GELU



@dataclass
class DispatchingNetConfig:
    """配送策略网络配置"""
    state_dim: int = 96        # 状态向量维度
    action_dim: int = 15       # 配送动作维度
    hidden_dim: int = 128      # 隐藏层维度
    lstm_layers: int = 2       # LSTM层数
    dropout: float = 0.1       # Dropout比率
    activation: ActivationType = ActivationType.GELU
    transformer_layers: int = 2  # Transformer层数
    transformer_heads: int = 4   # 注意力头数
    use_layer_norm: bool = True   # 是否使用层归一化


@dataclass
class NetworkConfig:
    """网络架构总配置"""
    meta_controller: MetaControllerNetConfig = field(default_factory=MetaControllerNetConfig)
    scheduling: SchedulingNetConfig = field(default_factory=SchedulingNetConfig)
    dispatching: DispatchingNetConfig = field(default_factory=DispatchingNetConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'meta_controller': self.meta_controller.__dict__,
            'scheduling': self.scheduling.__dict__,
            'dispatching': self.dispatching.__dict__
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'NetworkConfig':
        meta = MetaControllerNetConfig(**config_dict.get('meta_controller', {}))
        sched = SchedulingNetConfig(**config_dict.get('scheduling', {}))
        disp = DispatchingNetConfig(**config_dict.get('dispatching', {}))
        return cls(meta_controller=meta, scheduling=sched, dispatching=disp)


# === 训练配置相关类 ===
@dataclass
class BaseTrainingConfig:
    """基础训练配置"""
    learning_rate: float = 1e-4    # 学习率
    gamma: float = 0.99            # 折扣因子
    tau: float = 0.005             # 目标网络软更新系数
    batch_size: int = 64           # 训练批次大小
    buffer_size: int = 100000      # 经验回放缓冲区大小
    grad_clip: float = 0.5         # 梯度裁剪阈值

@dataclass
class MetaControllerTrainConfig(BaseTrainingConfig):
    """元控制器训练配置"""
    update_freq: int = 5           # 策略更新频率
    entropy_coef: float = 0.01     # 熵正则化系数
    value_loss_coef: float = 0.5   # 价值损失系数
    ppo_epochs: int = 10           # PPO更新轮数
    clip_param: float = 0.2        # PPO裁剪参数

@dataclass
class SchedulingTrainConfig(BaseTrainingConfig):
    """调度策略训练配置"""
    n_step_returns: int = 5        # n步回报
    priority_alpha: float = 0.6     # 优先经验回放alpha参数
    priority_beta: float = 0.4      # 优先经验回放beta参数
    dueling_network: bool = True    # 是否使用Dueling网络结构
    double_q: bool = True          # 是否使用Double DQN
    policy_update_freq: int = 5           # 策略更新频率
    value_loss_coef: float = 0.5   # 价值损失系数
    entropy_coef: float = 0.01     # 熵正则化系数

@dataclass
class DispatchingTrainConfig(BaseTrainingConfig):
    """配送策略训练配置"""
    sac_alpha: float = 0.2         # SAC温度参数
    auto_entropy_tuning: bool = True # 是否自动调整熵参数
    reward_scale: float = 1.0       # 奖励缩放因子
    q_update_steps: int = 1         # Q网络更新步数
    policy_update_freq: int = 2     # 策略更新频率
    

@dataclass
class TrainingConfig:
    """训练总配置"""
    meta_controller: MetaControllerTrainConfig = field(default_factory=MetaControllerTrainConfig)
    scheduling: SchedulingTrainConfig = field(default_factory=SchedulingTrainConfig)
    dispatching: DispatchingTrainConfig = field(default_factory=DispatchingTrainConfig)
    
    # 通用评估参数
    eval_episodes: int = 50         # 评估轮数
    save_interval: int = 100        # 模型保存间隔
    log_interval: int = 10          # 日志记录间隔
    max_episodes: int = 1000        # 最大训练轮数
    warmup_episodes: int = 10       # 预热轮数

    def to_dict(self) -> Dict[str, Any]:
        return {
            'meta_controller': self.meta_controller.__dict__,
            'scheduling': self.scheduling.__dict__,
            'dispatching': self.dispatching.__dict__,
            'eval_episodes': self.eval_episodes,
            'save_interval': self.save_interval,
            'log_interval': self.log_interval,
            'max_episodes': self.max_episodes,
            'warmup_episodes': self.warmup_episodes
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        meta = MetaControllerTrainConfig(**config_dict.get('meta_controller', {}))
        sched = SchedulingTrainConfig(**config_dict.get('scheduling', {}))
        disp = DispatchingTrainConfig(**config_dict.get('dispatching', {}))
        
        return cls(
            meta_controller=meta,
            scheduling=sched,
            dispatching=disp,
            eval_episodes=config_dict.get('eval_episodes', 50),
            save_interval=config_dict.get('save_interval', 100),
            log_interval=config_dict.get('log_interval', 10),
            max_episodes=config_dict.get('max_episodes', 1000),
            warmup_episodes=config_dict.get('warmup_episodes', 10)
        )

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

    # 新增参数
    max_weight: float = 1000.0     # 最大重量限制
    max_priority: int = 10         # 最大优先级
    urgent_threshold: int = 50     # 紧急阈值(分钟)
    max_machine_queue: int = 20    # 机器最大队列长度
    base_processing_time: int = 5  # 基础处理时间
    per_job_time: int = 2         # 每个作业增加的处理时间

     # 动态到达相关参数
    arrival_probability: float = 0.3    # 每个时间步新作业到达的概率
    arrival_batch_size: float = 1.5     # 每次到达的作业数量参数(泊松分布均值)
    min_operations: int = 2             # 最小工序数
    max_operations: int = 5             # 最大工序数
    min_processing_time: int = 2        # 最小加工时间
    max_processing_time: int = 10       # 最大加工时间
    due_window: int = 50               # 交期时间窗口
    max_priority: int = 10             # 最大优先级

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
    
    def save(self, filepath) -> None:
        """保存配置到文件"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath) -> 'Config':
        """从文件加载配置"""
        with open(filepath) as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)