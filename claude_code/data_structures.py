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
    due_date: int = 0 # 最终交付截止时间
    earliest_due_date: int = 0 # 最早交付截止时间
    delivery_requirements: List[Dict[str, Any]] = field(default_factory=list)  # 交付要求列表


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

