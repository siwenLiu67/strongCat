from dataclasses import dataclass, field
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum

class MachineStatus(Enum):
    """机器状态枚举"""
    IDLE = "idle"           # 空闲
    BUSY = "busy"          # 加工中
    SETUP = "setup"        # 装载准备
    MAINTENANCE = "maintenance"  # 维护中

@dataclass
class Machine:
    """机器实体类"""
    machine_id: int
    capabilities: List[int]  # 可加工的工序类型
    
    # 当前状态
    status: MachineStatus = MachineStatus.IDLE
    current_job: Optional[int] = None
    current_operation: Optional[int] = None
    remaining_time: float = 0.0
    total_busy_time: float = 0.0

    
    # 统计信息
    total_processing_time: float = 0.0
    total_idle_time: float = 0.0
    total_setup_time: float = 0.0
    job_history: List[Dict] = field(default_factory=list)
    
    def assign_job(self, job_id: int, operation_id: int, processing_time: float) -> None:
        """分配工作到机器"""
        self.current_job = job_id
        self.current_operation = operation_id
        self.remaining_time = processing_time
        self.status = MachineStatus.BUSY
        
    def complete_current_job(self, current_time: float) -> Optional[int]:
        """完成当前工作
        
        Returns:
            Optional[int]: 完成的工件ID
        """
        if self.current_job is not None:
            completed_job = self.current_job
            self.job_history.append({
                'job_id': self.current_job,
                'operation_id': self.current_operation,
                'complete_time': current_time
            })
            
            self.current_job = None
            self.current_operation = None
            self.remaining_time = 0
            self.status = MachineStatus.IDLE
            
            return completed_job
        return None
    
    def update_time(self, time_step: float) -> Optional[int]:
        """更新机器时间
        
        Returns:
            Optional[int]: 如果工作完成，返回工件ID
        """
        if self.status == MachineStatus.BUSY:
            self.remaining_time -= time_step
            self.total_processing_time += time_step
            
            if self.remaining_time <= 0:
                return self.complete_current_job(time_step)
                
        elif self.status == MachineStatus.IDLE:
            self.total_idle_time += time_step
            
        elif self.status == MachineStatus.SETUP:
            self.total_setup_time += time_step
            
        return None
    
    @property
    def utilization(self) -> float:
        """计算机器利用率"""
        total_time = self.total_processing_time + self.total_idle_time + self.total_setup_time
        return self.total_processing_time / max(total_time, 1e-6)
    

    
@dataclass
class Operation:
    """工序类"""
    operation_id: int
    available_machines: List[int]
    processing_times: Dict[int, int]  # machine_id -> processing_time
    status: str = 'waiting'
    complete_time: Optional[float] = None

@dataclass
class Job:
    """工件类"""
    job_id: int
    operations: List[Operation]
    distributor_id: int
    current_op: int = 0
    status: str = 'waiting'
    start_time: Optional[float] = None
    complete_time: Optional[float] = None
    arrival_time: Optional[float] = None

@dataclass
class DeliveryRequirement:
    """交付要求类"""
    distributor_id: int
    due_time: int
    ratio: float
    weight: float
    jobs: List[int]

@dataclass
class DistributorAssignment:
    """配送商分配类"""
    distributor_id: int
    assigned_jobs: List[int] = field(default_factory=list)
    delivery_requirements: List[DeliveryRequirement] = field(default_factory=list)