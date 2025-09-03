"""
动态优先规则启发式算法的环境适配器
参考ppo_environment_adaptor的复用方式，封装WarehouseEnvironment
实现事件驱动的启发式排序机制
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from environment import WarehouseEnvironment
from case_generator import FlexibleJobShopScenario
from config import Config
from dispatch_heuristic import DispatchHeuristic


class DynamicPriorityRuleEnvironmentAdapter:
    """动态优先规则启发式算法的环境适配器"""
    
    def __init__(self, config: Config, scenario: FlexibleJobShopScenario, rule_name: str = "EDD"):
        """初始化适配器
        
        Args:
            config: 配置对象
            scenario: 算例生成器实例
            rule_name: 优先规则名称
        """
        self.base_env = WarehouseEnvironment(config, scenario)
        self.scenario = scenario
        self.config = config
        self.rule_name = rule_name
        
        # 优先规则映射
        self.rule_mapping = {
            "EDD": self._earliest_due_date,
            "SPT": self._shortest_processing_time,
            "MST": self._minimum_slack_time,
            "CR": self._critical_ratio
        }
        
        if rule_name not in self.rule_mapping:
            raise ValueError(f"不支持的优先规则: {rule_name}。支持: {list(self.rule_mapping.keys())}")
        
        # 事件回调函数
        self.on_machine_idle_callbacks = []
        self.on_job_arrival_callbacks = []
        self.on_dispatch_ready_callbacks = []
        
        # 派发启发式算法
        self.dispatch_heuristic = DispatchHeuristic()
        
        # 调度记录
        self.schedule_records = []
        
        # 跟踪上一次的状态，用于检测变化
        self.last_idle_machines = set()
        self.last_available_jobs_count = 0
    
    def reset(self) -> Dict:
        """重置环境"""
        state = self.base_env.reset()
        
        # 重置跟踪状态
        self.last_idle_machines = set(m.machine_id for m in self.base_env.machines if m.status == 'waiting')
        self.last_available_jobs_count = len(self.base_env.available_jobs)
        self.schedule_records = []
        
        return state
    
    def step(self, action: Dict = {}) -> Tuple[Dict, float, bool, Dict, Dict]:
        """执行环境步进，返回状态、奖励、完成标志和信息"""
        # 启发式算法不接收外部动作，总是执行等待
        if not action:
            action = {'wait': {}}
            
        # base_env.step() 返回 (state, reward, done, info)
        state, reward, done, info = self.base_env.step(action)
        
        # 检测事件并触发回调
        self._detect_and_trigger_events()
        
        # 保持与PPO适配器兼容的返回格式，添加第五个空字典参数
        return state, reward, done, info, {}
    
    def _detect_and_trigger_events(self):
        """检测环境变化并触发相应事件"""
        # 检测机器空闲事件
        current_idle_machines = set(m.machine_id for m in self.base_env.machines if m.status == 'waiting')
        new_idle_machines = current_idle_machines - self.last_idle_machines
        
        if new_idle_machines:
            for callback in self.on_machine_idle_callbacks:
                callback(list(new_idle_machines))
        
        # 检测作业到达事件
        current_jobs_count = len(self.base_env.available_jobs)
        if current_jobs_count > self.last_available_jobs_count:
            new_jobs_count = current_jobs_count - self.last_available_jobs_count
            for callback in self.on_job_arrival_callbacks:
                callback(new_jobs_count)
        
        # 检测派发就绪事件（有已完成但未派发的作业）
        completed_undispatched = [j for j in self.base_env.completed_jobs 
                                if j not in self.base_env.dispatched_jobs]
        if completed_undispatched:
            for callback in self.on_dispatch_ready_callbacks:
                callback(len(completed_undispatched))
        
        # 更新跟踪状态
        self.last_idle_machines = current_idle_machines
        self.last_available_jobs_count = current_jobs_count
    
    def register_machine_idle_callback(self, callback: Callable[[List[int]], None]):
        """注册机器空闲事件回调"""
        self.on_machine_idle_callbacks.append(callback)
    
    def register_job_arrival_callback(self, callback: Callable[[int], None]):
        """注册作业到达事件回调"""
        self.on_job_arrival_callbacks.append(callback)
    
    def register_dispatch_ready_callback(self, callback: Callable[[int], None]):
        """注册派发就绪事件回调"""
        self.on_dispatch_ready_callbacks.append(callback)
    
    def apply_priority_rule_scheduling(self):
        """应用优先规则进行调度"""
        available_jobs = self._get_available_jobs()
        if not available_jobs:
            return
        
        # 应用优先规则排序
        sorted_jobs = self._apply_priority_rule(available_jobs)
        
        # 给所有空闲机器分配作业
        self._schedule_to_all_idle_machines(sorted_jobs)
    
    def apply_dispatch_heuristic(self):
        """应用派发启发式算法"""
        # 获取环境状态
        state = {
            'completed_jobs': self.base_env.completed_jobs,
            'config': self.base_env.config,
            't': self.base_env.t
        }
        
        # 使用派发启发式算法选择派发动作
        dispatch_action = self.dispatch_heuristic.select_action(state)
        
        # 执行派发动作
        if dispatch_action and 'dispatch' in dispatch_action:
            for batch_id, job_ids in dispatch_action['dispatch'].items():
                # 执行派发逻辑
                self._execute_dispatch(batch_id, job_ids)
    
    def _get_available_jobs(self) -> List:
        """获取当前可用的作业"""
        return [job for job in self.base_env.available_jobs if not self._is_job_completed(job)]
    
    def _apply_priority_rule(self, jobs: List) -> List:
        """应用优先规则对作业进行排序"""
        rule_func = self.rule_mapping[self.rule_name]
        return sorted(jobs, key=lambda job: rule_func(job, self.base_env.t))
    
    def _schedule_to_all_idle_machines(self, sorted_jobs: List):
        """给所有空闲机器分配作业"""
        if not sorted_jobs:
            return
        
        # 获取所有空闲机器
        idle_machines = [machine for machine in self.base_env.machines if machine.status == 'waiting']
        
        if not idle_machines:
            return
        
        # 为每个空闲机器尝试分配一个作业
        assigned_jobs = set()
        for machine in idle_machines:
            for job in sorted_jobs:
                if job.job_id in assigned_jobs:
                    continue
                
                # 检查作业是否有当前操作需要调度
                if job.current_operation >= len(job.operations):
                    continue
                
                operation_to_schedule = job.operations[job.current_operation]
                
                # 检查机器是否适合该操作
                if machine.machine_id in operation_to_schedule.available_machine_ids:
                    # 创建调度动作
                    schedule_action = {'schedule': {job.job_id: machine.machine_id}}
                    
                    # 执行调度动作
                    self.base_env.step(schedule_action)
                    
                    # 记录调度
                    schedule_record = {
                        'job_id': job.job_id,
                        'machine_id': machine.machine_id,
                        'time': self.base_env.t
                    }
                    self.schedule_records.append(schedule_record)
                    
                    assigned_jobs.add(job.job_id)
                    break
    
    def _execute_dispatch(self, batch_id: int, job_ids: List[int]):
        """执行派发操作"""
        # 创建派发动作
        dispatch_action = {'dispatch': {batch_id: job_ids}}
        
        # 执行派发动作
        self.base_env.step(dispatch_action)
    
    def _earliest_due_date(self, job, current_time: float) -> float:
        """最早截止日期规则"""
        return job.due_date
    
    def _shortest_processing_time(self, job, current_time: float) -> float:
        """最短处理时间规则"""
        remaining_processing_time = 0
        for op in job.operations:
            if job.current_operation <= job.operations.index(op):
                if op.processing_times:
                    remaining_processing_time += min(op.processing_times.values())
                else:
                    remaining_processing_time += 1
        return remaining_processing_time
    
    def _minimum_slack_time(self, job, current_time: float) -> float:
        """最小松弛时间规则"""
        remaining_processing_time = 0
        for op in job.operations:
            if job.current_operation <= job.operations.index(op):
                if op.processing_times:
                    remaining_processing_time += min(op.processing_times.values())
                else:
                    remaining_processing_time += 1
        
        slack_time = job.due_date - current_time - remaining_processing_time
        return slack_time
    
    def _critical_ratio(self, job, current_time: float) -> float:
        """临界比率规则"""
        remaining_processing_time = 0
        for op in job.operations:
            if job.current_operation <= job.operations.index(op):
                if op.processing_times:
                    remaining_processing_time += min(op.processing_times.values())
                else:
                    remaining_processing_time += 1
        
        time_remaining = job.due_date - current_time
        
        if time_remaining <= 0:
            return float('inf')
        
        if remaining_processing_time <= 0:
            return float('inf')
        
        return time_remaining / max(remaining_processing_time, 0.1)
    
    def _is_job_completed(self, job) -> bool:
        """检查作业是否已完成"""
        return job.status == 'completed' or job.status == 'dispatched' or job.current_operation >= len(job.operations)

    def get_schedule_records(self) -> List[Dict]:
        """返回调度记录"""
        return self.schedule_records

    @property
    def env(self):
        """返回底层的环境实例"""
        return self.base_env

    @property
    def jobs(self):
        """获取作业列表"""
        return self.base_env.available_jobs + self.base_env.completed_jobs
    
    @property
    def machines(self):
        """获取机器列表"""
        return self.base_env.machines
    
    @property
    def distributors(self):
        """获取配送商列表"""
        return self.base_env.distributors
    
    @property
    def completed_jobs(self):
        """获取已完成作业列表"""
        return self.base_env.completed_jobs
    
    @property
    def dispatched_jobs(self):
        """获取已派遣作业列表"""
        return self.base_env.dispatched_jobs
    
    @property
    def current_time(self):
        """获取当前时间"""
        return self.base_env.t
    
    @property
    def tardy_penalty(self):
        """获取延迟惩罚"""
        return self.base_env.tardy_penalty
    
    @property
    def total_weighted_tardiness(self):
        """获取总加权延迟时间"""
        return self.base_env.total_weighted_tardiness
    
    @property
    def dynamic_jobs_arrived(self):
        """获取已到达的动态作业数"""
        return self.base_env.dynamic_jobs_arrived
    
    @property
    def remaining_dynamic_jobs(self):
        """获取剩余动态作业数"""
        return self.base_env.remaining_dynamic_jobs
    
    @property
    def arrival_events(self):
        """获取到达事件记录"""
        return self.base_env.arrival_events
    
    @property
    def completion_times(self):
        """获取完成时间记录"""
        return self.base_env.completion_times
    
    @property
    def machine_utilization(self):
        """获取机器利用率统计"""
        return self.base_env.machine_utilization
    
    @property
    def initial_jobs(self):
        """获取初始作业列表"""
        return self.base_env.initial_jobs
    
    @property
    def available_jobs(self):
        """获取可用作业列表"""
        return self.base_env.available_jobs
    
    @property
    def done(self):
        """获取是否完成"""
        return self.base_env.done
    
    def get_dynamic_arrival_statistics(self):
        """获取动态到达统计信息"""
        return self.base_env.get_dynamic_arrival_statistics()
    
    def calculate_machine_utilization(self):
        """计算机器利用率"""
        return self.base_env.calculate_machine_utilization()
    
    def calculate_operation_progress_ratio(self):
        """计算作业进度比例"""
        return self.base_env.calculate_operation_progress_ratio()
    
    def calculate_machine_load_variance(self):
        """计算机器负载方差"""
        return self.base_env.calculate_machine_load_variance()


# 兼容性包装器
class DynamicPriorityRuleEnvironment:
    """兼容性包装器，保持与原始接口相同的API"""
    
    def __init__(self, scenario: FlexibleJobShopScenario, rule_name: str = "EDD"):
        """初始化兼容性环境"""
        config = Config()
        self.adapter = DynamicPriorityRuleEnvironmentAdapter(config, scenario, rule_name)
        self.scenario = scenario
        self.rule_name = rule_name
        
    def reset(self):
        """重置环境"""
        return self.adapter.reset()
    
    def step(self):
        """执行环境步进"""
        return self.adapter.step()
    
    @property
    def jobs(self):
        return self.adapter.jobs
    
    @property
    def machines(self):
        return self.adapter.machines
    
    @property
    def distributors(self):
        return self.adapter.distributors
    
    @property
    def completed_jobs(self):
        return self.adapter.completed_jobs
    
    @property
    def dispatched_jobs(self):
        return self.adapter.dispatched_jobs
    
    @property
    def current_time(self):
        return self.adapter.current_time
    
    @property
    def tardy_penalty(self):
        return self.adapter.tardy_penalty
    
    @property
    def total_weighted_tardiness(self):
        return self.adapter.total_weighted_tardiness
    
    @property
    def dynamic_jobs_arrived(self):
        return self.adapter.dynamic_jobs_arrived
    
    @property
    def remaining_dynamic_jobs(self):
        return self.adapter.remaining_dynamic_jobs
    
    @property
    def arrival_events(self):
        return self.adapter.arrival_events
    
    @property
    def completion_times(self):
        return self.adapter.completion_times
    
    @property
    def machine_utilization(self):
        return self.adapter.machine_utilization
    
    @property
    def initial_jobs(self):
        return self.adapter.initial_jobs
    
    @property
    def available_jobs(self):
        return self.adapter.available_jobs
    
    @property
    def done(self):
        return self.adapter.done
