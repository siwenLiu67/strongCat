import numpy as np
from typing import List, Tuple, Dict, Optional, Set
import copy
from enum import Enum
from abc import ABC, abstractmethod

class BatchStatus(Enum):
    """批次状态枚举"""
    WAITING = "waiting"        # 等待工件完成
    READY = "ready"           # 可以发运
    SHIPPED = "shipped"       # 已发运
    OVERDUE = "overdue"       # 逾期

class Job:
    """工件类"""
    def __init__(self, job_id: int, operations: List[Dict], distributor_id: int, priority: int = 0):
        """
        初始化工件
        
        Args:
            job_id: 工件ID
            operations: 工序列表
            distributor_id: 所属配送商ID
            priority: 优先级
        """
        self.job_id = job_id
        self.operations = operations
        self.num_operations = len(operations)
        self.current_operation = 0
        self.completed = False
        self.completion_times = {}
        self.distributor_id = distributor_id
        self.priority = priority
        self.finish_time = None  # 工件完工时间
        self.assigned_batch = None  # 分配到的批次ID
        self.shipped = False
    
    def get_current_operation(self):
        """获取当前工序信息"""
        if self.completed or self.current_operation >= self.num_operations:
            return None
        return self.operations[self.current_operation]
    
    def complete_operation(self, completion_time: float):
        """完成当前工序"""
        self.completion_times[self.current_operation] = completion_time
        self.current_operation += 1
        if self.current_operation >= self.num_operations:
            self.completed = True
            self.finish_time = completion_time
    
    def get_operation_completion_time(self, operation_id: int) -> float:
        """获取指定工序的完成时间"""
        return self.completion_times.get(operation_id, 0)
    
    def is_operation_ready(self, operation_id: int) -> bool:
        """检查指定工序是否可以开始"""
        return operation_id == self.current_operation and not self.completed
    
    def reset(self):
        """重置工件状态"""
        self.current_operation = 0
        self.completed = False
        self.completion_times = {}
        self.finish_time = None
        self.assigned_batch = None
        self.shipped = False

class Machine:
    """机器类"""
    def __init__(self, machine_id: int):
        self.machine_id = machine_id
        self.available_time = 0.0
        self.current_job = None
        self.schedule = []
    
    def is_available(self, time: float) -> bool:
        return self.available_time <= time
    
    def schedule_operation(self, job_id: int, operation_id: int, start_time: float, processing_time: float):
        completion_time = start_time + processing_time
        self.available_time = completion_time
        self.current_job = job_id
        
        self.schedule.append({
            'job_id': job_id,
            'operation_id': operation_id,
            'start_time': start_time,
            'completion_time': completion_time,
            'processing_time': processing_time
        })
        
        return completion_time
    
    def reset(self):
        self.available_time = 0.0
        self.current_job = None
        self.schedule = []

class Batch:
    """发运批次类"""
    def __init__(self, batch_id: int, distributor_id: int, due_date: float, capacity: int = None):
        """
        初始化批次
        
        Args:
            batch_id: 批次ID
            distributor_id: 配送商ID
            due_date: 截止时间
            capacity: 批次容量（可选）
        """
        self.batch_id = batch_id
        self.distributor_id = distributor_id
        self.due_date = due_date
        self.capacity = capacity
        self.jobs: Set[int] = set()  # 分配到此批次的工件ID
        self.status = BatchStatus.WAITING
        self.ship_time = None
        self.delay_penalty = 0.0
    
    def add_job(self, job_id: int) -> bool:
        """添加工件到批次"""
        if self.capacity and len(self.jobs) >= self.capacity:
            return False
        self.jobs.add(job_id)
        return True
    
    def remove_job(self, job_id: int):
        """从批次中移除工件"""
        self.jobs.discard(job_id)
    
    def can_ship(self, jobs_dict: Dict[int, Job]) -> bool:
        """检查是否可以发运（所有工件都已完成）"""
        if not self.jobs:
            return False
        return all(jobs_dict[job_id].completed for job_id in self.jobs)
    
    def ship(self, current_time: float):
        """发运批次"""
        self.ship_time = current_time
        self.status = BatchStatus.SHIPPED
        if current_time > self.due_date:
            self.status = BatchStatus.OVERDUE
            self.delay_penalty = current_time - self.due_date
    
    def reset(self):
        """重置批次状态"""
        self.jobs.clear()
        self.status = BatchStatus.WAITING
        self.ship_time = None
        self.delay_penalty = 0.0

class Distributor:
    """配送商类"""
    def __init__(self, distributor_id: int, name: str, due_dates: List[float]):
        """
        初始化配送商
        
        Args:
            distributor_id: 配送商ID
            name: 配送商名称
            due_dates: 3个批次的截止时间列表
        """
        self.distributor_id = distributor_id
        self.name = name
        self.job_ids: Set[int] = set()  # 所属工件ID集合
        
        # 创建3个批次
        self.batches = []
        for i, due_date in enumerate(due_dates):
            batch = Batch(
                batch_id=i,
                distributor_id=distributor_id,
                due_date=due_date
            )
            self.batches.append(batch)
    
    def add_job(self, job_id: int):
        """添加工件"""
        self.job_ids.add(job_id)
    
    def get_available_batches(self) -> List[Batch]:
        """获取可用的批次（未发运且未满）"""
        return [batch for batch in self.batches if batch.status == BatchStatus.WAITING]
    
    def get_ready_batches(self, jobs_dict: Dict[int, Job]) -> List[Batch]:
        """获取可以发运的批次"""
        ready_batches = []
        for batch in self.batches:
            if batch.status == BatchStatus.WAITING and batch.can_ship(jobs_dict):
                ready_batches.append(batch)
        return ready_batches
    
    def calculate_total_delay(self) -> float:
        """计算总延迟惩罚"""
        return sum(batch.delay_penalty for batch in self.batches)
    
    def reset(self):
        """重置配送商状态"""
        for batch in self.batches:
            batch.reset()

class Case:
    """算例类"""
    def __init__(self, name: str, jobs_data: List[Dict], distributors_data: List[Dict], num_machines: int):
        """
        初始化算例
        
        Args:
            name: 算例名称
            jobs_data: 工件数据列表，每个包含operations和distributor_id
            distributors_data: 配送商数据列表，每个包含name和due_dates
            num_machines: 机器数量
        """
        self.name = name
        self.num_machines = num_machines
        self.jobs_data = jobs_data
        self.distributors_data = distributors_data
        self.num_jobs = len(jobs_data)
        self.num_distributors = len(distributors_data)
        
        # 计算最大工序数
        self.max_operations = max(len(job_data['operations']) for job_data in jobs_data)
        
        # 验证数据完整性
        self._validate_data()
    
    def _validate_data(self):
        """验证算例数据的完整性"""
        distributor_ids = set(range(self.num_distributors))
        
        for job_id, job_data in enumerate(self.jobs_data):
            # 检查配送商ID有效性
            if job_data['distributor_id'] not in distributor_ids:
                raise ValueError(f"Invalid distributor_id {job_data['distributor_id']} for job {job_id}")
            
            # 检查工序数据
            for op_id, op_data in enumerate(job_data['operations']):
                machines = op_data['machines']
                times = op_data['times']
                
                if len(machines) != len(times):
                    raise ValueError(f"Job {job_id} Operation {op_id}: machines and times length mismatch")
                
                for machine_id in machines:
                    if machine_id < 0 or machine_id >= self.num_machines:
                        raise ValueError(f"Invalid machine ID {machine_id}")
        
        # 检查配送商数据
        for dist_data in self.distributors_data:
            if len(dist_data['due_dates']) != 3:
                raise ValueError("Each distributor must have exactly 3 due dates")
    
    def create_jobs(self) -> List[Job]:
        """创建工件实例"""
        jobs = []
        for job_id, job_data in enumerate(self.jobs_data):
            job = Job(
                job_id=job_id,
                operations=job_data['operations'],
                distributor_id=job_data['distributor_id'],
                priority=job_data.get('priority', 0)
            )
            jobs.append(job)
        return jobs
    
    def create_machines(self) -> List[Machine]:
        """创建机器实例"""
        return [Machine(i) for i in range(self.num_machines)]
    
    def create_distributors(self) -> List[Distributor]:
        """创建配送商实例"""
        distributors = []
        for dist_id, dist_data in enumerate(self.distributors_data):
            distributor = Distributor(
                distributor_id=dist_id,
                name=dist_data['name'],
                due_dates=dist_data['due_dates']
            )
            distributors.append(distributor)
        
        # 为配送商分配工件
        for job_id, job_data in enumerate(self.jobs_data):
            dist_id = job_data['distributor_id']
            distributors[dist_id].add_job(job_id)
        
        return distributors

class HierarchicalAction:
    """层次化动作基类"""
    pass

class SchedulingAction(HierarchicalAction):
    """调度动作：选择(工件, 工序, 机器)"""
    def __init__(self, job_id: int, operation_id: int, machine_id: int):
        self.job_id = job_id
        self.operation_id = operation_id
        self.machine_id = machine_id
    
    def __repr__(self):
        return f"Schedule(J{self.job_id}_O{self.operation_id}_M{self.machine_id})"

class BatchingAction(HierarchicalAction):
    """批次分配动作：选择(工件, 批次)"""
    def __init__(self, job_id: int, batch_id: int):
        self.job_id = job_id
        self.batch_id = batch_id
    
    def __repr__(self):
        return f"Batch(J{self.job_id}_B{self.batch_id})"

class ShippingAction(HierarchicalAction):
    """发运动作：选择发运哪个批次"""
    def __init__(self, distributor_id: int, batch_id: int):
        self.distributor_id = distributor_id
        self.batch_id = batch_id
    
    def __repr__(self):
        return f"Ship(D{self.distributor_id}_B{self.batch_id})"

class DecisionLevel(Enum):
    """决策层级"""
    SCHEDULING = "scheduling"    # 生产调度层
    BATCHING = "batching"       # 批次分配层
    SHIPPING = "shipping"       # 发运决策层

class FJSPEnvironment:
    """带配送商场景的层次化FJSP强化学习环境"""
    
    def __init__(self, case: Case):
        self.case = case
        self.num_jobs = case.num_jobs
        self.num_machines = case.num_machines
        self.num_distributors = case.num_distributors
        self.max_operations = case.max_operations
        
        # 层次化决策相关
        self.current_decision_level = DecisionLevel.SCHEDULING
        self.scheduling_completed = False
        
        # 状态空间维度（根据决策层级不同而不同）
        self.scheduling_state_dim = self._calculate_scheduling_state_dim()
        self.batching_state_dim = self._calculate_batching_state_dim()
        self.shipping_state_dim = self._calculate_shipping_state_dim()
        
        self.reset()
    
    def _calculate_scheduling_state_dim(self) -> int:
        """计算调度层状态空间维度"""
        return (
            1 +  # 当前时间
            self.num_machines * 2 +  # 机器状态
            self.num_jobs * 3 +  # 工件状态(当前工序, 完成状态, 配送商ID)
            self.num_jobs * self.max_operations  # 操作状态
        )
    
    def _calculate_batching_state_dim(self) -> int:
        """计算批次分配层状态空间维度"""
        return (
            1 +  # 当前时间
            self.num_jobs * 4 +  # 工件状态(完成时间, 配送商ID, 是否已分配, 优先级)
            self.num_distributors * 3 * 3  # 每个配送商3个批次的状态(due_date, 当前工件数, 状态)
        )
    
    def _calculate_shipping_state_dim(self) -> int:
        """计算发运层状态空间维度"""
        return (
            1 +  # 当前时间
            self.num_distributors * 3 * 4  # 每个配送商3个批次(due_date, 工件数, 是否就绪, 延迟)
        )
    
    def reset(self):
        """重置环境"""
        self.current_time = 0.0
        self.jobs = self.case.create_jobs()
        self.machines = self.case.create_machines()
        self.distributors = self.case.create_distributors()
        
        # 创建工件字典便于查找
        self.jobs_dict = {job.job_id: job for job in self.jobs}
        
        # 重置层次化状态
        self.current_decision_level = DecisionLevel.SCHEDULING
        self.scheduling_completed = False
        
        # 统计信息
        self.total_operations = sum(job.num_operations for job in self.jobs)
        self.completed_operations = 0
        self.schedule_history = []
        self.batch_history = []
        self.shipping_history = []
        
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """根据当前决策层级获取状态"""
        if self.current_decision_level == DecisionLevel.SCHEDULING:
            return self._get_scheduling_state()
        elif self.current_decision_level == DecisionLevel.BATCHING:
            return self._get_batching_state()
        else:  # SHIPPING
            return self._get_shipping_state()
    
    def _get_scheduling_state(self) -> np.ndarray:
        """获取调度层状态"""
        state = []
        
        # 当前时间
        state.append(self.current_time)
        
        # 机器状态
        for machine in self.machines:
            state.append(machine.available_time)
            state.append(machine.current_job if machine.current_job is not None else -1)
        
        # 工件状态
        for job in self.jobs:
            state.append(job.current_operation)
            state.append(1 if job.completed else 0)
            state.append(job.distributor_id)
        
        # 操作状态
        for job in self.jobs:
            for op_id in range(self.max_operations):
                if op_id < job.num_operations:
                    if op_id < job.current_operation:
                        state.append(2)  # 已完成
                    elif op_id == job.current_operation and not job.completed:
                        state.append(1)  # 可调度
                    else:
                        state.append(0)  # 等待中
                else:
                    state.append(-1)  # 不存在
        
        return np.array(state, dtype=np.float32)
    
    def _get_batching_state(self) -> np.ndarray:
        """获取批次分配层状态"""
        state = []
        
        # 当前时间
        state.append(self.current_time)
        
        # 工件状态
        for job in self.jobs:
            state.append(job.finish_time if job.finish_time else 0)
            state.append(job.distributor_id)
            state.append(1 if job.assigned_batch is not None else 0)
            state.append(job.priority)
        
        # 批次状态
        for distributor in self.distributors:
            for batch in distributor.batches:
                state.append(batch.due_date)
                state.append(len(batch.jobs))
                state.append(batch.status.value == BatchStatus.WAITING.value)
        
        return np.array(state, dtype=np.float32)
    
    def _get_shipping_state(self) -> np.ndarray:
        """获取发运层状态"""
        state = []
        
        # 当前时间
        state.append(self.current_time)
        
        # 批次状态
        for distributor in self.distributors:
            for batch in distributor.batches:
                state.append(batch.due_date)
                state.append(len(batch.jobs))
                state.append(1 if batch.can_ship(self.jobs_dict) else 0)
                delay = max(0, self.current_time - batch.due_date) if batch.jobs else 0
                state.append(delay)
        
        return np.array(state, dtype=np.float32)
    
    def get_valid_actions(self) -> List[HierarchicalAction]:
        """根据当前决策层级获取有效动作"""
        if self.current_decision_level == DecisionLevel.SCHEDULING:
            return self._get_valid_scheduling_actions()
        elif self.current_decision_level == DecisionLevel.BATCHING:
            return self._get_valid_batching_actions()
        else:  # SHIPPING
            return self._get_valid_shipping_actions()
    
    def _get_valid_scheduling_actions(self) -> List[SchedulingAction]:
        """获取有效的调度动作"""
        valid_actions = []
        
        for job in self.jobs:
            if job.completed:
                continue
            
            current_op = job.get_current_operation()
            if current_op is None:
                continue
            
            available_machines = current_op['machines']
            for machine_id in available_machines:
                action = SchedulingAction(job.job_id, job.current_operation, machine_id)
                valid_actions.append(action)
        
        return valid_actions
    
    def _get_valid_batching_actions(self) -> List[BatchingAction]:
        """获取有效的批次分配动作"""
        valid_actions = []
        
        # 找到已完成但未分配批次的工件
        for job in self.jobs:
            if job.completed and job.assigned_batch is None:
                distributor = self.distributors[job.distributor_id]
                available_batches = distributor.get_available_batches()
                
                for batch in available_batches:
                    action = BatchingAction(job.job_id, batch.batch_id)
                    valid_actions.append(action)
        
        return valid_actions
    
    def _get_valid_shipping_actions(self) -> List[ShippingAction]:
        """获取有效的发运动作"""
        valid_actions = []
        
        for distributor in self.distributors:
            ready_batches = distributor.get_ready_batches(self.jobs_dict)
            for batch in ready_batches:
                action = ShippingAction(distributor.distributor_id, batch.batch_id)
                valid_actions.append(action)
        
        return valid_actions
    
    def step(self, action: HierarchicalAction) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行层次化动作"""
        if isinstance(action, SchedulingAction):
            return self._step_scheduling(action)
        elif isinstance(action, BatchingAction):
            return self._step_batching(action)
        elif isinstance(action, ShippingAction):
            return self._step_shipping(action)
        else:
            raise ValueError(f"Unknown action type: {type(action)}")
    
    def _step_scheduling(self, action: SchedulingAction) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行调度动作"""
        # 验证动作有效性
        valid_actions = self._get_valid_scheduling_actions()
        if not any(a.job_id == action.job_id and a.operation_id == action.operation_id 
                  and a.machine_id == action.machine_id for a in valid_actions):
            return self.get_state(), -100.0, False, {"error": "Invalid scheduling action"}
        
        # 执行调度逻辑（与原来类似）
        job = self.jobs[action.job_id]
        machine = self.machines[action.machine_id]
        current_op = job.get_current_operation()
        
        # 获取加工时间
        machine_idx = current_op['machines'].index(action.machine_id)
        processing_time = current_op['times'][machine_idx]
        
        # 计算开始时间
        machine_ready_time = machine.available_time
        if action.operation_id > 0:
            job_ready_time = job.get_operation_completion_time(action.operation_id - 1)
        else:
            job_ready_time = 0.0
        
        start_time = max(machine_ready_time, job_ready_time)
        completion_time = machine.schedule_operation(action.job_id, action.operation_id, start_time, processing_time)
        
        # 完成工序
        job.complete_operation(completion_time)
        self.completed_operations += 1
        self.current_time = max(self.current_time, completion_time)
        
        # 记录历史
        self.schedule_history.append({
            'job_id': action.job_id,
            'operation_id': action.operation_id,
            'machine_id': action.machine_id,
            'start_time': start_time,
            'completion_time': completion_time,
            'processing_time': processing_time
        })
        
        # 检查是否完成所有调度
        if all(job.completed for job in self.jobs):
            self.scheduling_completed = True
            self.current_decision_level = DecisionLevel.BATCHING
        
        reward = self._calculate_scheduling_reward(completion_time)
        info = {
            'level': 'scheduling',
            'action': str(action),
            'completion_time': completion_time,
            'level_done': self.scheduling_completed
        }
        
        return self.get_state(), reward, False, info
    
    def _step_batching(self, action: BatchingAction) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行批次分配动作"""
        # 验证动作有效性
        valid_actions = self._get_valid_batching_actions()
        if not any(a.job_id == action.job_id and a.batch_id == action.batch_id for a in valid_actions):
            return self.get_state(), -50.0, False, {"error": "Invalid batching action"}
        
        # 执行批次分配
        job = self.jobs[action.job_id]
        distributor = self.distributors[job.distributor_id]
        batch = distributor.batches[action.batch_id]
        
        # 分配工件到批次
        if batch.add_job(action.job_id):
            job.assigned_batch = action.batch_id
            
            self.batch_history.append({
                'job_id': action.job_id,
                'batch_id': action.batch_id,
                'distributor_id': job.distributor_id,
                'assignment_time': self.current_time
            })
        
        # 检查是否完成所有批次分配
        batching_done = all(job.assigned_batch is not None for job in self.jobs if job.completed)
        if batching_done:
            self.current_decision_level = DecisionLevel.SHIPPING
        
        reward = self._calculate_batching_reward(job, batch)
        info = {
            'level': 'batching',
            'action': str(action),
            'level_done': batching_done
        }
        
        return self.get_state(), reward, False, info
    
    def _step_shipping(self, action: ShippingAction) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行发运动作"""
        # 验证动作有效性
        valid_actions = self._get_valid_shipping_actions()
        if not any(a.distributor_id == action.distributor_id and a.batch_id == action.batch_id 
                  for a in valid_actions):
            return self.get_state(), -50.0, False, {"error": "Invalid shipping action"}
        
        # 执行发运
        distributor = self.distributors[action.distributor_id]
        batch = distributor.batches[action.batch_id]
        
        # 更新发运时间
        max_completion_time = max(self.jobs_dict[job_id].finish_time for job_id in batch.jobs)
        ship_time = max(self.current_time, max_completion_time)
        
        batch.ship(ship_time)
        self.current_time = max(self.current_time, ship_time)
        
        # 标记工件为已发运
        for job_id in batch.jobs:
            self.jobs_dict[job_id].shipped = True
        
        self.shipping_history.append({
            'distributor_id': action.distributor_id,
            'batch_id': action.batch_id,
            'ship_time': ship_time,
            'due_date': batch.due_date,
            'delay': batch.delay_penalty,
            'job_count': len(batch.jobs)
        })
        
        # 检查是否完成所有发运
        shipping_done = all(batch.status in [BatchStatus.SHIPPED, BatchStatus.OVERDUE] 
                           for distributor in self.distributors 
                           for batch in distributor.batches 
                           if batch.jobs)
        
        reward = self._calculate_shipping_reward(batch)
        info = {
            'level': 'shipping',
            'action': str(action),
            'ship_time': ship_time,
            'delay': batch.delay_penalty,
            'level_done': shipping_done
        }
        
        return self.get_state(), reward, shipping_done, info
    
    def _calculate_scheduling_reward(self, completion_time: float) -> float:
        """计算调度层奖励"""
        base_reward = 10.0
        time_penalty = -completion_time * 0.01
        progress_reward = (self.completed_operations / self.total_operations) * 20.0
        
        if self.scheduling_completed:
            makespan = max(machine.available_time for machine in self.machines)
            makespan_reward = 100.0 / (makespan + 1.0)
            return base_reward + progress_reward + makespan_reward
        
        return base_reward + time_penalty + progress_reward
    
    def _calculate_batching_reward(self, job: Job, batch: Batch) -> float:
        """计算批次分配层奖励"""
        base_reward = 5.0
        
        # 根据截止时间紧迫性给予不同奖励
        urgency = max(0, batch.due_date - job.finish_time)
        urgency_reward = 10.0 / (urgency + 1.0)
        
        # 优先级奖励
        priority_reward = job.priority * 2.0
        
        return base_reward + urgency_reward + priority_reward
    
    def _calculate_shipping_reward(self, batch: Batch) -> float:
        """计算发运层奖励"""
        base_reward = 20.0
        
        # 准时发运奖励
        if batch.status == BatchStatus.SHIPPED:
            ontime_reward = 50.0
        else:  # OVERDUE
            ontime_reward = -batch.delay_penalty * 10.0
        
        # 批次大小奖励
        batch_size_reward = len(batch.jobs) * 2.0
        
        return base_reward + ontime_reward + batch
    
    