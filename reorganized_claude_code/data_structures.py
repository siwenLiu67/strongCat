"""
数据结构模块
定义仓储-配送环境中的所有实体类
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class Operation:
    """工序实体类"""
    operation_id: int
    job_id: int
    available_machine_ids: List[int]  # 可用机器ID列表
    processing_times: Dict[int, int]  # 机器ID到处理时间的映射
    status: str = "waiting"  # 状态: waiting, processing, completed
    completed_time: float = 0.0  # 完成时间
    
    def get_processing_time(self, machine_id: int) -> int:
        """获取指定机器的处理时间"""
        return self.processing_times.get(machine_id, 0)


@dataclass
class Job:
    """工件实体类"""
    job_id: int
    operations: List[Operation]
    amount: int  # 工件数量
    distributor_id: int
    current_operation: int = 0  # 当前工序索引
    status: str = "waiting"  # 状态: waiting, processing, completed, dispatching, dispatched
    completed_time: float = 0.0  # 完成时间
    dispatched_time: float = 0.0  # 完成配送时间
    due_date: int = 0  # 最终交付截止时间
    earliest_due_date: int = 0  # 最早交付截止时间
    delivery_requirements: List[Dict[str, Any]] = field(default_factory=list)  # 交付要求列表
    is_urgent: bool = False  # 是否为紧急工件
    arrival_time: float = 0.0  # 到达时间
    assigned_delivery_window: Dict[str, Any] = field(default_factory=dict)  # 分配的交付窗口信息
    job_type: str = "initial"  # 作业类型: initial, dynamic
    assigned_machine: Optional[int] = None  # 分配的机器ID
    
    def get_current_operation(self) -> Optional[Operation]:
        """获取当前工序"""
        if 0 <= self.current_operation < len(self.operations):
            return self.operations[self.current_operation]
        return None
    
    def is_completed(self) -> bool:
        """检查作业是否完成"""
        return self.current_operation >= len(self.operations)
    
    def is_dispatched(self) -> bool:
        """检查作业是否已配送"""
        return self.status == "dispatched"


@dataclass
class DeliveryRequirement:
    """交付要求类"""
    distributor_id: int
    due_times: List[int]  # 交付截止时间列表
    ratios: List[float]  # 交付比例列表
    weights: List[float]  # 交付权重列表
    
    def validate(self) -> bool:
        """验证交付要求数据"""
        if len(self.due_times) != len(self.ratios) or len(self.ratios) != len(self.weights):
            return False
        if sum(self.ratios) != 1.0:
            return False
        return True


@dataclass
class Machine:
    """机器实体类"""
    machine_id: int
    current_job: Optional[Job] = None  # 当前处理的作业
    remaining_time: float = 0.0  # 剩余处理时间
    total_busy_time: float = 0.0  # 总忙碌时间
    total_idle_time: float = 0.0  # 总空闲时间
    processed_jobs: List[int] = field(default_factory=list)  # 已处理的工件ID列表
    status: str = "idle"  # 状态: idle, busy
    
    def assign_job(self, job: Job, processing_time: float):
        """分配工件到机器"""
        self.current_job = job
        self.remaining_time = processing_time
        self.status = "busy"
        self.processed_jobs.append(job.job_id)
        self.total_busy_time += processing_time
    
    def complete_job(self):
        """完成当前作业"""
        self.current_job = None
        self.remaining_time = 0.0
        self.status = "idle"
    
    def is_available(self) -> bool:
        """检查机器是否可用"""
        return self.status == "idle"


@dataclass
class Distributor:
    """分发者实体类"""
    distributor_id: int
    delivery_requirements: Optional[DeliveryRequirement] = None
    assigned_jobs: List[int] = field(default_factory=list)  # 属于这个配送商的作业ID列表
    total_amount: int = 0  # 总配送数量
    completed_batches: Dict[int, List[Job]] = field(default_factory=dict)  # 完成的批次
    completed_times: Dict[int, float] = field(default_factory=dict)  # batchID到完成时间的映射
    overdue_times: Dict[int, int] = field(default_factory=dict)  # batchID到逾期数量的映射
    status: str = "waiting"  # 状态: waiting, completed
    dispatching_time: int = 1  # 配送时间
    
    def add_job(self, job: Job):
        """添加作业到配送商"""
        self.assigned_jobs.append(job.job_id)
        self.total_amount += job.amount
    
    def remove_job(self, job: Job):
        """从配送商移除作业"""
        if job.job_id in self.assigned_jobs:
            self.assigned_jobs.remove(job.job_id)
            self.total_amount -= job.amount


@dataclass
class EnvironmentState:
    """环境状态快照"""
    time: int
    machine_states: List[Dict[str, Any]]
    job_states: List[Dict[str, Any]]
    distributor_states: List[Dict[str, Any]]
    completed_jobs_count: int
    dispatched_jobs_count: int
    available_jobs_count: int
    remaining_dynamic_jobs: int
    dynamic_jobs_arrived: int
    machine_utilization: float
    operation_progress_ratio: float
    machine_load_variance: float
    total_weighted_tardiness: float
    tardy_penalty: float


@dataclass
class Action:
    """动作基类"""
    action_type: str  # wait, schedule, dispatch


@dataclass
class WaitAction(Action):
    """等待动作"""
    def __init__(self):
        super().__init__("wait")


@dataclass
class ScheduleAction(Action):
    """调度动作"""
    schedule: Dict[int, int]  # job_id -> machine_id
    
    def __init__(self, schedule: Dict[int, int]):
        super().__init__("schedule")
        self.schedule = schedule


@dataclass
class DispatchAction(Action):
    """配送动作"""
    dispatch: Dict[int, List[int]]  # distributor_id -> job_ids
    
    def __init__(self, dispatch: Dict[int, List[int]]):
        super().__init__("dispatch")
        self.dispatch = dispatch


@dataclass
class ExperimentResult:
    """实验结果"""
    algorithm_name: str
    instance_name: str
    total_reward: float
    total_time_steps: int
    completed_jobs: int
    dispatched_jobs: int
    machine_utilization: float
    total_weighted_tardiness: float
    tardy_penalty: float
    convergence_data: List[float]
    execution_time: float
    parameters: Dict[str, Any]
