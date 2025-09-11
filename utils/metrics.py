"""
通用的实验指标数据类
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ExperimentMetrics:
    """实验指标数据类，用于记录实验过程中的各种指标"""
    def __init__(self,
                 episode_reward: float = 0.0,
                 episode_length: int = 0,
                 makespan: float = 0.0,
                 running_time: float = 0.0,
                 tardy_penalty: float = 0.0,
                 total_tardiness: float = 0.0,
                 meta_loss: float = 0.0,
                 machine_utilization: float = 0.0):
        self.episode_reward = episode_reward
        self.episode_length = episode_length
        self.makespan = makespan
        self.running_time = running_time
        self.tardy_penalty = tardy_penalty
        self.total_tardiness = total_tardiness
        self.meta_loss = meta_loss
        self.machine_utilization = machine_utilization

    @property
    def objective_value(self) -> float:
        """计算目标函数值"""
        return self.episode_reward + self.tardy_penalty + self.total_tardiness
    
    def to_dict(self) -> Dict[str, Any]:
        """将指标转换为字典形式"""
        return {
            'episode_reward': self.episode_reward,
            'episode_length': self.episode_length,
            'makespan': self.makespan,
            'running_time': self.running_time,
            'tardy_penalty': self.tardy_penalty,
            'total_tardiness': self.total_tardiness,
            'meta_loss': self.meta_loss,
            'machine_utilization': self.machine_utilization,
            'objective_value': self.objective_value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentMetrics':
        """从字典创建指标对象"""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})
