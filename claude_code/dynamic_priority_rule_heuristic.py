"""
动态优先规则启发式算法 - 支持动态环境下的实时重新排序
每次决策时都对所有未调度的工件重新进行启发式排序
集成派发启发式算法，实现完整的加工和派发解决方案
使用环境适配器复用WarehouseEnvironment，参考ppo_environment_adaptor的方式
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from dynamic_priority_rule_environment_adapter import DynamicPriorityRuleEnvironmentAdapter
from environment import WarehouseEnvironment
from data_structures import Job, Operation, Machine, Distributor
from config import Config
from case_generator import FlexibleJobShopScenario
from dispatch_heuristic import DispatchHeuristic


class DynamicPriorityRuleHeuristicSolver:
    """
    动态优先规则启发式求解器，支持动态环境下的实时重新排序
    每次决策时都对所有未调度的工件重新进行启发式排序
    集成派发启发式算法，实现完整的加工和派发解决方案
    使用环境适配器复用WarehouseEnvironment
    """
    
    def __init__(self, env: WarehouseEnvironment, config, rule_name: str = "EDD"):
        """
        初始化求解器
        
        Args:
            env: 仓库环境实例
            rule_name: 优先规则名称，可选值: "EDD", "SPT", "MST", "CR"
        """
        self.config = config
        self.rule_name = rule_name
        
        # 使用环境适配器而不是直接使用环境
        self.env_adapter = DynamicPriorityRuleEnvironmentAdapter(config, env.case, rule_name)
        self.env_adapter.register_machine_idle_callback(self._on_machine_idle)
        self.env_adapter.register_job_arrival_callback(self._on_job_arrival)

        self.rule_mapping = {
            "EDD": self._earliest_due_date,
            "SPT": self._shortest_processing_time,
            "MST": self._minimum_slack_time,
            "CR": self._critical_ratio
        }
        
        if rule_name not in self.rule_mapping:
            raise ValueError(f"不支持的优先规则: {rule_name}。支持: {list(self.rule_mapping.keys())}")
        
        # 初始化调度信息跟踪
        self.schedule_records = []  # 存储所有调度记录
        self.current_time = 0
        self.dispatch_heuristic = DispatchHeuristic()  # 派发启发式算法
    
    def solve(self) -> Dict[str, Any]:
        """
        求解调度问题，返回DQN兼容格式的结果
        包含完整的加工和派发流程
        
        Returns:
            DQN兼容的结果字典，包含stats、env、additional_metrics
        """
        start_time = time.time()
        
        # 重置环境适配器到初始状态
        self.env_adapter.reset()
        self.current_time = 0
        
        # 执行调度（加工和派发）
        self._schedule_all_jobs()
        
        # 计算性能指标
        stats = self._calculate_performance_metrics()
        
        # 计算机器利用率
        machine_utilization = self._calculate_machine_utilization()
        
        # 计算解决时间
        solve_time = time.time() - start_time
        
        # 返回DQN兼容格式的结果
        return {
            "stats": stats,
            "env": self.env_adapter.env,  # 返回实际的环境实例
            "additional_metrics": {
                "solve_time": solve_time,
                "machine_utilization": machine_utilization,
                "algorithm_name": f"DynamicPriorityRule_{self.rule_name}"
            }
        }
    
    def _schedule_all_jobs(self):
        """
        通过事件驱动的方式调度所有作业
        循环推进环境，直到满足终止条件
        """
        done = False
        while not done:
            # 推进环境，这将自动处理事件并调用回调函数
            _, _, done, _, _ = self.env_adapter.step({})  # 传递一个空的action
            
            # 派发逻辑可以在每个时间步后执行
            self._dispatch_completed_jobs()
            
            # 更新当前时间
            self.current_time = self.env_adapter.current_time

        # 收集调度记录
        self.schedule_records = self.env_adapter.get_schedule_records()
        
        # 终止检查
        max_time_steps = getattr(self.env_adapter.config, 'max_time_steps', 1000)
        if self.current_time > max_time_steps:
            print(f"⚠️ 警告：达到最大时间步 {max_time_steps}，但仍有作业未完成或未派发")
            print(f"详细状态: 可用作业={len(self.env_adapter.available_jobs)}, 已完成作业={len(self.env_adapter.completed_jobs)}, 已派发作业={len(self.env_adapter.dispatched_jobs)}")
            print(f"剩余动态作业: {self.env_adapter.remaining_dynamic_jobs}")
            
            for job in self.env_adapter.available_jobs + self.env_adapter.completed_jobs:
                if not self._is_job_completed(job):
                    print(f"作业 {job.job_id} 状态: {job.status}, 当前工序: {job.current_operation}/{len(job.operations)}")
    
    def _dispatch_completed_jobs(self):
        """派发已完成的作业"""
        # 获取环境状态
        state = {
            'completed_jobs': self.env_adapter.completed_jobs,
            'config': self.env_adapter.config,
            't': self.current_time
        }
        
        # 使用派发启发式算法选择派发动作
        dispatch_action = self.dispatch_heuristic.select_action(state)
        
        # 执行派发动作
        if dispatch_action and 'dispatch' in dispatch_action:
            for batch_id, job_ids in dispatch_action['dispatch'].items():
                # 执行派发逻辑
                self._execute_dispatch(batch_id, job_ids)
    
    def _execute_dispatch(self, batch_id: int, job_ids: List[int]):
        """执行派发操作，参考environment.py中的派发逻辑"""
        # 只处理已完成且未派发的作业
        dispatch_jobs = [job for job in self.env_adapter.completed_jobs 
                        if job.job_id in job_ids and job.status == 'completed']
        
        if dispatch_jobs:
            # 设置配送时间（参考environment.py中的逻辑）
            BASE_DELIVERY_TIME = 1  # 基础配送时间
            PER_JOB_TIME = 0        # 每多一个工件增加的配送时间
            batch_size = len(dispatch_jobs)
            delivery_time = BASE_DELIVERY_TIME + PER_JOB_TIME * batch_size
            
            # 更新作业状态为配送中
            for job in dispatch_jobs:
                job.status = 'dispatching'
                job.dispatch_remaining_time = delivery_time
                job.dispatch_start_time = self.current_time
                print(f"作业 {job.job_id} 开始配送，批次ID: {batch_id}，配送时间: {delivery_time}")
            
            # 推进配送中的作业
            self._update_dispatching_jobs()
    
    def _update_dispatching_jobs(self):
        """推进所有配送中的作业，配送完成后更新状态和时间"""
        for job in self.env_adapter.completed_jobs:
            if getattr(job, 'status', None) == 'dispatching':
                job.dispatch_remaining_time -= 1
                if job.dispatch_remaining_time <= 0:
                    job.status = 'dispatched'
                    job.dispatched_time = self.current_time
                    # 将作业从completed_jobs移动到dispatched_jobs
                    if job in self.env_adapter.completed_jobs:
                        self.env_adapter.completed_jobs.remove(job)
                    if job not in self.env_adapter.dispatched_jobs:
                        self.env_adapter.dispatched_jobs.append(job)
                    print(f"作业 {job.job_id} 配送完成，时间: {self.current_time}")
    
    def _find_job_by_id(self, job_id: int) -> Optional[Job]:
        """根据ID查找作业"""
        for job in self.env_adapter.available_jobs + self.env_adapter.completed_jobs:
            if job.job_id == job_id:
                return job
        return None
    
    def _all_jobs_dispatched(self) -> bool:
        """检查所有作业是否都已派发"""
        # 检查所有已完成的作业是否都已派发
        for job in self.env_adapter.completed_jobs:
            if job.status != 'dispatched':
                return False
        return True
    
    def _get_available_jobs(self, current_time: float) -> List[Job]:
        """获取当前时间可用的作业（已到达且未完成的作业）"""
        available_jobs = []
        
        for job in self.env_adapter.available_jobs:
            # 检查作业是否已到达
            if not self._is_job_completed(job):
                available_jobs.append(job)
        
        return available_jobs
    
    def _apply_priority_rule(self, jobs: List[Job], current_time: float) -> List[Job]:
        """应用优先规则对作业进行排序"""
        rule_func = self.rule_mapping[self.rule_name]
        return sorted(jobs, key=lambda job: rule_func(job, current_time))
    
    def _earliest_due_date(self, job: Job, current_time: float) -> float:
        """最早截止日期规则"""
        return job.due_date
    
    def _shortest_processing_time(self, job: Job, current_time: float) -> float:
        """最短处理时间规则"""
        remaining_processing_time = 0
        for op in job.operations:
            # 检查工序是否已完成
            if job.current_operation <= job.operations.index(op):
                # 获取最小处理时间
                if op.processing_times:
                    remaining_processing_time += min(op.processing_times.values())
                else:
                    remaining_processing_time += 1  # 默认值
        return remaining_processing_time
    
    def _minimum_slack_time(self, job: Job, current_time: float) -> float:
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
    
    def _critical_ratio(self, job: Job, current_time: float) -> float:
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
            return float('inf')  # 已经延迟，最高优先级
        
        if remaining_processing_time <= 0:
            return float('inf')  # 没有剩余处理时间
        
        return time_remaining / max(remaining_processing_time, 0.1)
    
    def _schedule_to_all_idle_machines(self, sorted_jobs: List[Job], current_time: float):
        """给所有空闲机器分配作业"""
        if not sorted_jobs:
            return
        
        # 获取所有空闲机器
        idle_machines = [machine for machine in self.env_adapter.machines if machine.status == 'waiting']
        
        if not idle_machines:
            return
        
        # 为每个空闲机器尝试分配一个作业
        assigned_jobs = set()  # 记录已分配的作业，避免重复分配
        for machine in idle_machines:
            # 找到适合该机器的最高优先级作业
            for job in sorted_jobs:
                if job.job_id in assigned_jobs:
                    continue  # 跳过已分配的作业
                
                # 检查作业是否有当前操作需要调度
                if job.current_operation >= len(job.operations):
                    continue  # 作业已完成所有工序
                
                operation_to_schedule = job.operations[job.current_operation]
                
                # 检查机器是否适合该操作
                if machine.machine_id in operation_to_schedule.available_machine_ids:
                    # 分配作业到机器
                    processing_time = operation_to_schedule.processing_times.get(machine.machine_id, 1)
                    
                    # 安排操作到机器
                    start_time = max(current_time, machine.remaining_time if machine.status == 'busy' else 0)
                    end_time = start_time + processing_time
                    
                    # 更新机器状态
                    machine.status = 'busy'
                    machine.current_job = job.job_id
                    machine.remaining_time = processing_time
                    
                    # 更新作业状态
                    job.status = 'processing'
                    job.current_operation += 1
                    
                    # 如果作业所有工序都已完成，更新状态为completed
                    if job.current_operation >= len(job.operations):
                        job.status = 'completed'
                        job.completed_time = end_time
                        # 将作业从未完成列表移动到完成列表
                        if job in self.env_adapter.available_jobs:
                            self.env_adapter.available_jobs.remove(job)
                            self.env_adapter.completed_jobs.append(job)
                    
                    # 记录调度信息
                    schedule_record = {
                        'job_id': job.job_id,
                        'operation_index': job.current_operation - 1,
                        'machine_id': machine.machine_id,
                        'start_time': start_time,
                        'end_time': end_time,
                        'processing_time': processing_time
                    }
                    job_type = getattr(job, 'job_type', 'initial')
                    print(f"调度记录: {schedule_record}, 作业类型: {job_type}")
                    self.schedule_records.append(schedule_record)
                    
                    # 特别记录动态作业的调度
                    if job_type == 'dynamic':
                        print(f"动态作业 {job.job_id} 被调度到机器 {machine.machine_id}")
                    
                    assigned_jobs.add(job.job_id)
                    break  # 为这台机器分配了一个作业，继续下一台机器
    
    def _on_machine_idle(self, machine_ids: List[int]):
        """机器空闲时的回调函数"""
        current_time = self.env_adapter.current_time
        for machine_id in machine_ids:
            print(f"事件：机器 {machine_id} 在时间 {current_time} 空闲，触发重新调度")
        self._trigger_reschedule(current_time)

    def _on_job_arrival(self, new_jobs_count: int):
        """作业到达时的回调函数"""
        current_time = self.env_adapter.current_time
        print(f"事件：{new_jobs_count} 个新作业在时间 {current_time} 到达，触发重新调度")
        self._trigger_reschedule(current_time)

    def _trigger_reschedule(self, current_time: float):
        """
        触发重新调度
        获取所有可用作业，排序并分配给所有空闲机器
        """
        available_jobs = self._get_available_jobs(current_time)
        if available_jobs:
            sorted_jobs = self._apply_priority_rule(available_jobs, current_time)
            self._schedule_to_all_idle_machines(sorted_jobs, current_time)
    
    def _find_available_machine(self, operation: Operation, current_time: float) -> Optional[Machine]:
        """找到可用的机器"""
        available_machines = []
        
        for machine in self.env_adapter.machines:
            if machine.machine_id in operation.available_machine_ids:
                # 检查机器是否空闲
                if machine.status == 'waiting':
                    available_machines.append(machine)
        
        if available_machines:
            # 选择处理时间最短的机器（SPT规则）
            return min(available_machines, key=lambda m: operation.processing_times.get(m.machine_id, float('inf')))
        
        return None
    
    def _get_next_event_time(self, current_time: float) -> float:
        """获取下一个事件时间（作业到达或机器空闲）"""
        next_times = []
        
        # 检查机器完成时间
        for machine in self.env_adapter.machines:
            if machine.status == 'busy':
                completion_time = current_time + machine.remaining_time
                if completion_time > current_time:
                    next_times.append(completion_time)
        
        return min(next_times) if next_times else current_time + 1
    
    def _update_job_states(self):
        """更新作业状态，将已完成的作业移动到完成列表"""
        for job in self.env_adapter.available_jobs[:]:  # 使用切片复制避免修改迭代中的列表
            if job.status == 'completed':
                if job not in self.env_adapter.completed_jobs:
                    self.env_adapter.completed_jobs.append(job)
                    job.completed_time = self.current_time
                # 从可用作业中移除已完成的作业
                if job in self.env_adapter.available_jobs:
                    self.env_adapter.available_jobs.remove(job)
    
    def _check_termination(self) -> bool:
        """检查终止条件，参考environment.py中的逻辑"""
        # 计算已到达的总作业数量（初始作业 + 已到达的动态作业）
        total_arrived_jobs = len(self.env_adapter.initial_jobs) + self.env_adapter.dynamic_jobs_arrived
        
        # 终止条件：所有已到达的作业都已派发且没有剩余动态作业
        # 或者达到最大时间步但动态作业没有到达（避免无限等待）
        return (len(self.env_adapter.dispatched_jobs) >= total_arrived_jobs and 
                self.env_adapter.remaining_dynamic_jobs <= 0)
    
    def _update_machine_states(self):
        """更新所有忙碌机器的状态，减少剩余时间，完成操作"""
        for machine in self.env_adapter.machines:
            if machine.status == 'busy' and machine.remaining_time > 0:
                machine.remaining_time -= 1
                
                # 如果机器完成当前操作
                if machine.remaining_time <= 0:
                    machine.status = 'waiting'
                    machine.current_job = -1
                    print(f"机器 {machine.machine_id} 完成操作，当前时间: {self.current_time}")
    
    def _get_next_decision_time(self, current_time: float) -> float:
        """获取下一个决策点时间"""
        # 推进到下一个事件时间（机器空闲、作业到达、配送完成）
        next_event_time = self._get_next_event_time(current_time)
        
        # 同时检查配送中的作业，找到下一个配送完成时间
        next_dispatch_time = float('inf')
        for job in self.env_adapter.completed_jobs:
            if getattr(job, 'status', None) == 'dispatching':
                dispatch_remaining = getattr(job, 'dispatch_remaining_time', 0)
                if dispatch_remaining > 0:
                    next_dispatch_time = min(next_dispatch_time, current_time + dispatch_remaining)
        
        # 取最小的事件时间，确保至少推进1个时间单位
        next_time = min(next_event_time, next_dispatch_time, current_time + 1)
        
        # 确保时间步正常推进，避免卡在某个时间点
        if next_time <= current_time:
            next_time = current_time + 1
            
        return next_time
    
    def _is_job_completed(self, job: Job) -> bool:
        """检查作业是否已完成"""
        # 检查作业状态或是否所有工序都已完成
        if job.status == 'completed' or job.status == 'dispatched':
            return True
        
        # 检查current_operation是否等于工序数量
        if job.current_operation >= len(job.operations):
            return True
        
        return False
    
    def _all_jobs_completed(self) -> bool:
        """检查所有作业是否已完成（只检查已到达的作业）"""
        # 只检查已到达的作业，不包括未到达的动态作业
        arrived_jobs = [job for job in self.env_adapter.available_jobs + self.env_adapter.completed_jobs 
                       if job.arrival_time <= self.current_time]
        return all(self._is_job_completed(job) for job in arrived_jobs)
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """计算性能指标"""
        makespan = self._calculate_makespan()
        total_tardiness = self._calculate_total_tardiness()
        total_delivery_time = self._calculate_total_delivery_time()
        on_time_delivery_rate = self._calculate_on_time_delivery_rate()
        objective_value = self._calculate_objective_value()
        tardy_penalty = self._calculate_tardy_penalty()
        total_weighted_tardiness = self._calculate_total_weighted_tardiness()
        
        return {
            "makespan": makespan,
            "total_tardiness": total_tardiness,
            "total_delivery_time": total_delivery_time,
            "on_time_delivery_rate": on_time_delivery_rate,
            "objective_value": objective_value,
            "tardy_penalty": tardy_penalty,
            "total_weighted_tardiness": total_weighted_tardiness
        }
    
    def _calculate_makespan(self) -> float:
        """计算最大完成时间"""
        if not self.schedule_records:
            return 0
        
        # 找到所有作业的最后完成时间
        job_completion_times = {}
        for record in self.schedule_records:
            job_id = record['job_id']
            end_time = record['end_time']
            if job_id not in job_completion_times or end_time > job_completion_times[job_id]:
                job_completion_times[job_id] = end_time
        
        return max(job_completion_times.values()) if job_completion_times else 0
    
    def _calculate_total_tardiness(self) -> float:
        """计算总延迟时间"""
        if not self.schedule_records:
            return 0
        
        total_tardiness = 0
        
        # 找到每个作业的最后完成时间
        job_completion_times = {}
        for record in self.schedule_records:
            job_id = record['job_id']
            end_time = record['end_time']
            if job_id not in job_completion_times or end_time > job_completion_times[job_id]:
                job_completion_times[job_id] = end_time
        
        # 计算每个作业的延迟时间
        for job in self.env_adapter.available_jobs + self.env_adapter.completed_jobs:
            if job.job_id in job_completion_times:
                completion_time = job_completion_times[job.job_id]
                tardiness = max(0, completion_time - job.due_date)
                total_tardiness += tardiness
        
        return total_tardiness
    
    def _calculate_total_delivery_time(self) -> float:
        """计算总交付时间"""
        total_delivery_time = 0
        for job in self.env_adapter.available_jobs + self.env_adapter.completed_jobs:
            # 使用调度记录来计算完成时间
            job_records = [r for r in self.schedule_records if r['job_id'] == job.job_id]
            if job_records:
                completion_time = max(record['end_time'] for record in job_records)
                total_delivery_time += completion_time
        return total_delivery_time
    
    def _calculate_on_time_delivery_rate(self) -> float:
        """计算准时交付率"""
        on_time_count = 0
        total_jobs = len(self.env_adapter.available_jobs + self.env_adapter.completed_jobs)
        
        for job in self.env_adapter.available_jobs + self.env_adapter.completed_jobs:
            # 使用调度记录来计算完成时间
            job_records = [r for r in self.schedule_records if r['job_id'] == job.job_id]
            if job_records:
                completion_time = max(record['end_time'] for record in job_records)
                if completion_time <= job.due_date:
                    on_time_count += 1
        
        return on_time_count / total_jobs if total_jobs > 0 else 0
    
    def _calculate_objective_value(self) -> float:
        """计算目标函数值"""
        # 这里使用总加权延迟时间作为目标函数
        return self._calculate_total_weighted_tardiness()
    
    def _calculate_tardy_penalty(self) -> float:
        """计算延迟惩罚"""
        # 简单实现：使用总延迟时间
        return self._calculate_total_tardiness()
    
    def _calculate_total_weighted_tardiness(self) -> float:
        """计算总加权延迟时间"""
        total_weighted_tardiness = 0
        for job in self.env_adapter.available_jobs + self.env_adapter.completed_jobs:
            # 使用调度记录来计算完成时间
            job_records = [r for r in self.schedule_records if r['job_id'] == job.job_id]
            if job_records:
                completion_time = max(record['end_time'] for record in job_records)
                tardiness = max(0, completion_time - job.due_date)
                weight = getattr(job, 'weight', 1.0)
                total_weighted_tardiness += weight * tardiness
        return total_weighted_tardiness

    def _calculate_machine_utilization(self) -> Dict[int, float]:
        """计算机器利用率"""
        machine_utilization = {}
        makespan = self._calculate_makespan()
        
        if makespan == 0:
            return {machine.machine_id: 0.0 for machine in self.env_adapter.machines}
        
        for machine in self.env_adapter.machines:
            busy_time = 0
            # 统计该机器上所有操作的加工时间
            for record in self.schedule_records:
                if record['machine_id'] == machine.machine_id:
                    busy_time += record['processing_time']
            
            utilization = busy_time / makespan if makespan > 0 else 0
            machine_utilization[machine.machine_id] = utilization
        
        return machine_utilization


def run_dynamic_priority_rule_heuristic(env_config: Dict, rule_name: str = "EDD") -> Dict[str, Any]:
    """
    运行动态优先规则启发式算法的便捷函数
    
    Args:
        env_config: 环境配置字典
        rule_name: 优先规则名称
        
    Returns:
        DQN兼容的结果字典
    """
    # 创建环境实例
    env = WarehouseEnvironment(**env_config)
    
    # 创建求解器
    solver = DynamicPriorityRuleHeuristicSolver(env, rule_name)
    
    # 求解并返回结果
    return solver.solve()


def run_dynamic_priority_rule_experiment(config, case, seed, **kwargs):
    """
    批量实验统一入口，供批量运行器调用
    参考ppo_model中的run_ppo_experiment函数格式
    
    Args:
        config: 配置对象
        case: 算例对象
        seed: 随机种子
        **kwargs: 其他参数
        
    Returns:
        与run_ppo_experiment兼容的结果字典
    """
    # 设置随机种子
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    
    # 创建环境实例
    env = WarehouseEnvironment(config, case)
    
    # 从kwargs获取规则名称，默认为EDD
    rule_name = kwargs.get('rule_name', 'EDD')
    
    # 创建求解器
    solver = DynamicPriorityRuleHeuristicSolver(env, config, rule_name)
    
    # 求解并获取结果
    result = solver.solve()
    
    # 返回与PPO实验兼容的格式
    return result
