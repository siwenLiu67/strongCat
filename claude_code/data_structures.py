from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import random
import numpy as np

@dataclass
class Operation:
    """工序实体类"""
    operation_id: int
    job_id: int
    available_machine_ids: List[int]  # 可用机器ID列表
    processing_times: Dict[int, int]  # 机器ID到处理时间的映射
    """机器ID到处理时间的映射"""
    status: str = "waiting"  # 状态: pending, in_progress, completed
    completed_time: float = 0.0  # 完成时间


@dataclass
class Job:
    """工件实体类"""
    job_id: int
    operations: List[Operation]
    distributor_id: int
    current_operation: int = 0  # 当前工序索引
    status: str = "waiting"  # 状态: waiting, processing, completed,dispatching, dispatched
    completed_time: float = 0.0  # 完成时间
    dispatched_time: float = 0.0 # 完成配送时间


@dataclass
class DeliveryRequirement:
    """交付要求类"""
    distributor_id: int
    due_times: List[int]  # 交付截止时间列表
    ratios: List[float]  # 分发者ID到交付比例的映射
    weights: List[float]  # 分发者ID到交付权重的映射
    

@dataclass
class Machine:
    """机器实体类"""
    machine_id: int

    # 时间相关
    current_job: int 
    remaining_time: float 
    total_busy_time: float  # 总忙碌时间
    total_idle_time: float  # 总空闲时间

    processed_jobs: List[int]  # 已处理的工件ID列表

    status: str = "waiting"  # 状态: waiting, busy

    def assign_job(self, job_id: int, processing_time: float):
        """分配工件到机器"""
        self.current_job = job_id
        self.remaining_time = processing_time
        self.status = "busy"
        self.processed_jobs.append(job_id)
        self.total_busy_time += processing_time


@dataclass
class Distributor:
    """分发者实体类"""
    distributor_id: int
    delivery_requirements: DeliveryRequirement
    assigned_jobs: List[int] # 属于这个配送商的作业ID列表

    # 和配送相关的属性
    completed_batches: Dict[int, List[Job]]
    completed_times: Dict[int, float]  # batchID到完成时间的映射
    overdue_times: Dict[int, int]  # batchID到逾期数量的映射

    status: str = "waiting"  # 状态: waiting, completed


class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.buffer = []
        self.position = 0
        
    def add(self, state: Any, action: Any, reward: float, next_state: Any, done: bool):
        """添加经验到缓冲区"""
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.buffer_size
        
    def sample(self, batch_size: int) -> Tuple:
        """从缓冲区随机采样一个batch"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
        
    def __len__(self) -> int:
        """返回当前缓冲区大小"""
        return len(self.buffer)
