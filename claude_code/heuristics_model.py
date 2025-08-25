"""
集成柔性作业车间调度与派遣问题的启发式算法求解器
包含4种典型的启发式算法：
1. 优先规则算法 (Priority Rule Algorithm)
2. 贪婪构造算法 (Greedy Construction Algorithm) 
3. 局部搜索算法 (Local Search Algorithm)
4. 遗传算法 (Genetic Algorithm)
"""

import numpy as np
import random
import time
import copy
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import heapq

from case_generator import FlexibleJobShopScenario
from config import Config
from data_structures import Job, Operation, Machine, Distributor


@dataclass
class ScheduleResult:
    """调度结果数据结构"""
    job_schedules: Dict[int, List[Dict]]  # {job_id: [{'operation': int, 'machine': int, 'start': int, 'end': int}]}
    job_completion_times: Dict[int, int]  # {job_id: completion_time}
    job_tardiness: Dict[int, int]  # {job_id: tardiness}
    makespan: int
    total_tardiness: int
    machine_utilization: Dict[int, float]  # {machine_id: utilization_rate}


@dataclass
class DispatchResult:
    """派遣结果数据结构"""
    batches: Dict[int, Dict]  # {batch_id: {'jobs': [], 'dispatch_time': int, 'distributor': int}}
    total_delivery_time: int
    delivery_cost: float
    on_time_delivery_rate: float


@dataclass
class SolutionResult:
    """完整解决方案结果"""
    schedule_result: ScheduleResult
    dispatch_result: DispatchResult
    total_objective: float
    solve_time: float
    algorithm_name: str


class PriorityRuleAlgorithm:
    """
    优先规则算法
    基于作业和机器的优先级规则进行调度和派遣
    """
    
    def __init__(self, scenario: FlexibleJobShopScenario):
        self.scenario = scenario
        self.jobs = scenario.jobs
        self.machines = scenario.machines
        self.distributors = scenario.distributors
        
    def solve(self, job_priority_rule: str = "EDD", 
              machine_selection_rule: str = "SPT") -> SolutionResult:
        """
        使用优先规则求解
        
        Args:
            job_priority_rule: 作业优先规则 ("EDD", "SPT", "LPT", "CR")
            machine_selection_rule: 机器选择规则 ("SPT", "LPT", "RANDOM")
        """
        start_time = time.time()
        
        # 阶段1: 车间调度
        schedule_result = self._schedule_jobs(job_priority_rule, machine_selection_rule)
        
        # 阶段2: 派遣决策
        dispatch_result = self._dispatch_jobs(schedule_result)
        
        # 计算总目标值
        total_objective = (schedule_result.total_tardiness + 
                          dispatch_result.total_delivery_time * 0.1)
        
        solve_time = time.time() - start_time
        
        return SolutionResult(
            schedule_result=schedule_result,
            dispatch_result=dispatch_result,
            total_objective=total_objective,
            solve_time=solve_time,
            algorithm_name=f"Priority_Rule_{job_priority_rule}_{machine_selection_rule}"
        )
    
    def _schedule_jobs(self, job_priority_rule: str, machine_selection_rule: str) -> ScheduleResult:
        """使用优先规则进行作业调度"""
        # 初始化
        job_schedules = {job.job_id: [] for job in self.jobs}
        job_completion_times = {}
        job_tardiness = {}
        machine_busy_until = {m.machine_id: 0 for m in self.machines}
        
        # 创建待调度工序队列
        ready_operations = []
        job_current_op = {job.job_id: 0 for job in self.jobs}
        job_last_completion = {job.job_id: getattr(job, 'arrival_time', 0) for job in self.jobs}
        
        # 初始化第一个工序
        for job in self.jobs:
            if job.operations:
                ready_operations.append((job.job_id, 0))
        
        # 按优先规则排序作业
        ready_operations = self._sort_operations_by_priority(ready_operations, job_priority_rule)
        
        # 调度循环
        while ready_operations:
            # 取优先级最高的工序
            job_id, op_idx = ready_operations.pop(0)
            job = next(j for j in self.jobs if j.job_id == job_id)
            operation = job.operations[op_idx]
            
            # 选择机器
            selected_machine, start_time, processing_time = self._select_machine(
                operation, machine_busy_until, job_last_completion[job_id], machine_selection_rule
            )
            
            if selected_machine is not None:
                end_time = start_time + processing_time
                
                # 记录调度结果
                job_schedules[job_id].append({
                    'operation': op_idx,
                    'machine': selected_machine,
                    'start': start_time,
                    'end': end_time,
                    'processing_time': processing_time
                })
                
                # 更新状态
                machine_busy_until[selected_machine] = end_time
                job_last_completion[job_id] = end_time
                
                # 检查是否有下一个工序
                if op_idx + 1 < len(job.operations):
                    # 将下一个工序加入队列
                    next_op = (job_id, op_idx + 1)
                    ready_operations.append(next_op)
                    # 重新排序
                    ready_operations = self._sort_operations_by_priority(ready_operations, job_priority_rule)
                else:
                    # 作业完成
                    job_completion_times[job_id] = end_time
                    due_date = getattr(job, 'due_date', 1000)
                    job_tardiness[job_id] = max(0, end_time - due_date)
        
        # 计算性能指标
        makespan = max(job_completion_times.values()) if job_completion_times else 0
        total_tardiness = sum(job_tardiness.values())
        
        # 计算机器利用率
        machine_utilization = {}
        for machine in self.machines:
            busy_time = sum(
                schedule['processing_time'] 
                for schedules in job_schedules.values()
                for schedule in schedules
                if schedule['machine'] == machine.machine_id
            )
            machine_utilization[machine.machine_id] = busy_time / max(makespan, 1)
        
        return ScheduleResult(
            job_schedules=job_schedules,
            job_completion_times=job_completion_times,
            job_tardiness=job_tardiness,
            makespan=makespan,
            total_tardiness=total_tardiness,
            machine_utilization=machine_utilization
        )
    
    def _sort_operations_by_priority(self, operations: List[Tuple], rule: str) -> List[Tuple]:
        """根据优先规则排序工序"""
        if rule == "EDD":  # Earliest Due Date
            return sorted(operations, key=lambda x: getattr(
                next(j for j in self.jobs if j.job_id == x[0]), 'due_date', 1000
            ))
        elif rule == "SPT":  # Shortest Processing Time
            return sorted(operations, key=lambda x: self._get_min_processing_time(x[0], x[1]))
        elif rule == "LPT":  # Longest Processing Time
            return sorted(operations, key=lambda x: self._get_min_processing_time(x[0], x[1]), reverse=True)
        elif rule == "CR":  # Critical Ratio
            return sorted(operations, key=lambda x: self._calculate_critical_ratio(x[0], x[1]))
        else:
            return operations
    
    def _get_min_processing_time(self, job_id: int, op_idx: int) -> int:
        """获取工序的最小处理时间"""
        job = next(j for j in self.jobs if j.job_id == job_id)
        operation = job.operations[op_idx]
        processing_times = getattr(operation, 'processing_times', {})
        return min(processing_times.values()) if processing_times else 1
    
    def _calculate_critical_ratio(self, job_id: int, op_idx: int) -> float:
        """计算临界比率"""
        job = next(j for j in self.jobs if j.job_id == job_id)
        due_date = getattr(job, 'due_date', 1000)
        
        # 计算剩余处理时间
        remaining_time = sum(
            min(op.processing_times.values()) if getattr(op, 'processing_times', {}) else 1
            for i, op in enumerate(job.operations[op_idx:])
        )
        
        # 临界比率 = (截止时间 - 当前时间) / 剩余处理时间
        current_time = 0  # 简化处理
        return (due_date - current_time) / max(remaining_time, 1)
    
    def _select_machine(self, operation: Operation, machine_busy_until: Dict, 
                       earliest_start: int, rule: str) -> Tuple[Optional[int], int, int]:
        """选择机器"""
        eligible_machines = getattr(operation, 'available_machine_ids', [])
        processing_times = getattr(operation, 'processing_times', {})
        
        if not eligible_machines or not processing_times:
            return None, 0, 0
        
        candidates = []
        for machine_id in eligible_machines:
            if machine_id in processing_times:
                proc_time = processing_times[machine_id]
                start_time = max(earliest_start, machine_busy_until[machine_id])
                candidates.append((machine_id, start_time, proc_time))
        
        if not candidates:
            return None, 0, 0
        
        if rule == "SPT":  # 最短处理时间
            selected = min(candidates, key=lambda x: x[2])
        elif rule == "LPT":  # 最长处理时间
            selected = max(candidates, key=lambda x: x[2])
        elif rule == "EFT":  # 最早完成时间
            selected = min(candidates, key=lambda x: x[1] + x[2])
        else:  # RANDOM
            selected = random.choice(candidates)
        
        return selected
    
    def _dispatch_jobs(self, schedule_result: ScheduleResult) -> DispatchResult:
        """使用简单规则进行派遣决策"""
        batches = {}
        batch_id = 0
        
        # 按配送商分组作业
        distributor_jobs = defaultdict(list)
        for job in self.jobs:
            distributor_id = getattr(job, 'distributor_id', 0)
            completion_time = schedule_result.job_completion_times.get(job.job_id, 0)
            distributor_jobs[distributor_id].append((job.job_id, completion_time))
        
        total_delivery_time = 0
        on_time_count = 0
        total_jobs = len(self.jobs)
        
        # 为每个配送商创建批次
        for distributor_id, jobs in distributor_jobs.items():
            # 按完成时间排序
            jobs.sort(key=lambda x: x[1])
            
            # 简单批次策略：每3个作业一批
            batch_size = 3
            for i in range(0, len(jobs), batch_size):
                batch_jobs = jobs[i:i + batch_size]
                job_ids = [job_id for job_id, _ in batch_jobs]
                completion_times = [comp_time for _, comp_time in batch_jobs]
                
                # 派遣时间为批次中最晚完成的作业时间
                dispatch_time = max(completion_times)
                
                batches[batch_id] = {
                    'jobs': job_ids,
                    'dispatch_time': dispatch_time,
                    'distributor': distributor_id,
                    'completion_times': dict(batch_jobs)
                }
                
                total_delivery_time += dispatch_time
                
                # 计算按时交付
                for job_id in job_ids:
                    job = next(j for j in self.jobs if j.job_id == job_id)
                    due_date = getattr(job, 'due_date', 1000)
                    if dispatch_time <= due_date:
                        on_time_count += 1
                
                batch_id += 1
        
        on_time_delivery_rate = on_time_count / max(total_jobs, 1)
        delivery_cost = total_delivery_time * 0.1  # 简化成本模型
        
        return DispatchResult(
            batches=batches,
            total_delivery_time=total_delivery_time,
            delivery_cost=delivery_cost,
            on_time_delivery_rate=on_time_delivery_rate
        )


class GreedyConstructionAlgorithm:
    """
    贪婪构造算法
    逐步构造解决方案，每步选择当前最优的决策
    """
    
    def __init__(self, scenario: FlexibleJobShopScenario):
        self.scenario = scenario
        self.jobs = scenario.jobs
        self.machines = scenario.machines
        self.distributors = scenario.distributors
    
    def solve(self) -> SolutionResult:
        """使用贪婪构造算法求解"""
        start_time = time.time()
        
        # 同时考虑调度和派遣的贪婪构造
        schedule_result, dispatch_result = self._greedy_construction()
        
        # 计算总目标值
        total_objective = (schedule_result.total_tardiness + 
                          dispatch_result.total_delivery_time * 0.1)
        
        solve_time = time.time() - start_time
        
        return SolutionResult(
            schedule_result=schedule_result,
            dispatch_result=dispatch_result,
            total_objective=total_objective,
            solve_time=solve_time,
            algorithm_name="Greedy_Construction"
        )
    
    def _greedy_construction(self) -> Tuple[ScheduleResult, DispatchResult]:
        """贪婪构造过程"""
        # 初始化
        job_schedules = {job.job_id: [] for job in self.jobs}
        job_completion_times = {}
        machine_busy_until = {m.machine_id: 0 for m in self.machines}
        
        # 待调度工序队列
        ready_operations = []
        job_current_op = {job.job_id: 0 for job in self.jobs}
        job_last_completion = {job.job_id: getattr(job, 'arrival_time', 0) for job in self.jobs}
        
        # 初始化
        for job in self.jobs:
            if job.operations:
                ready_operations.append((job.job_id, 0))
        
        # 贪婪构造调度
        while ready_operations:
            best_choice = None
            best_value = float('inf')
            
            # 评估所有可能的(工序, 机器)组合
            for job_id, op_idx in ready_operations:
                job = next(j for j in self.jobs if j.job_id == job_id)
                operation = job.operations[op_idx]
                eligible_machines = getattr(operation, 'available_machine_ids', [])
                processing_times = getattr(operation, 'processing_times', {})
                
                for machine_id in eligible_machines:
                    if machine_id in processing_times:
                        # 计算此选择的评估值
                        evaluation = self._evaluate_choice(
                            job_id, op_idx, machine_id, 
                            machine_busy_until, job_last_completion[job_id]
                        )
                        
                        if evaluation < best_value:
                            best_value = evaluation
                            best_choice = (job_id, op_idx, machine_id)
            
            if best_choice is None:
                break
            
            # 执行最佳选择
            job_id, op_idx, machine_id = best_choice
            job = next(j for j in self.jobs if j.job_id == job_id)
            operation = job.operations[op_idx]
            processing_times = getattr(operation, 'processing_times', {})
            
            start_time = max(job_last_completion[job_id], machine_busy_until[machine_id])
            processing_time = processing_times[machine_id]
            end_time = start_time + processing_time
            
            # 更新调度
            job_schedules[job_id].append({
                'operation': op_idx,
                'machine': machine_id,
                'start': start_time,
                'end': end_time,
                'processing_time': processing_time
            })
            
            machine_busy_until[machine_id] = end_time
            job_last_completion[job_id] = end_time
            
            # 移除已调度的工序
            ready_operations.remove((job_id, op_idx))
            
            # 添加后续工序
            if op_idx + 1 < len(job.operations):
                ready_operations.append((job_id, op_idx + 1))
            else:
                job_completion_times[job_id] = end_time
        
        # 构造调度结果
        makespan = max(job_completion_times.values()) if job_completion_times else 0
        job_tardiness = {}
        for job in self.jobs:
            completion_time = job_completion_times.get(job.job_id, 0)
            due_date = getattr(job, 'due_date', 1000)
            job_tardiness[job.job_id] = max(0, completion_time - due_date)
        
        total_tardiness = sum(job_tardiness.values())
        
        # 计算机器利用率
        machine_utilization = {}
        for machine in self.machines:
            busy_time = sum(
                schedule['processing_time'] 
                for schedules in job_schedules.values()
                for schedule in schedules
                if schedule['machine'] == machine.machine_id
            )
            machine_utilization[machine.machine_id] = busy_time / max(makespan, 1)
        
        schedule_result = ScheduleResult(
            job_schedules=job_schedules,
            job_completion_times=job_completion_times,
            job_tardiness=job_tardiness,
            makespan=makespan,
            total_tardiness=total_tardiness,
            machine_utilization=machine_utilization
        )
        
        # 贪婪派遣构造
        dispatch_result = self._greedy_dispatch_construction(schedule_result)
        
        return schedule_result, dispatch_result
    
    def _evaluate_choice(self, job_id: int, op_idx: int, machine_id: int,
                        machine_busy_until: Dict, job_ready_time: int) -> float:
        """评估(作业, 工序, 机器)组合的贪婪值"""
        job = next(j for j in self.jobs if j.job_id == job_id)
        operation = job.operations[op_idx]
        processing_times = getattr(operation, 'processing_times', {})
        
        start_time = max(job_ready_time, machine_busy_until[machine_id])
        processing_time = processing_times.get(machine_id, 1)
        completion_time = start_time + processing_time
        
        # 多目标评估函数
        due_date = getattr(job, 'due_date', 1000)
        tardiness = max(0, completion_time - due_date)
        
        # 考虑的因素：
        # 1. 延误惩罚
        # 2. 机器空闲时间
        # 3. 处理时间
        machine_idle = max(0, start_time - machine_busy_until[machine_id])
        
        evaluation = (
            tardiness * 2.0 +           # 延误权重
            machine_idle * 0.1 +        # 机器空闲权重
            processing_time * 0.5       # 处理时间权重
        )
        
        return evaluation
    
    def _greedy_dispatch_construction(self, schedule_result: ScheduleResult) -> DispatchResult:
        """贪婪派遣构造"""
        batches = {}
        batch_id = 0
        
        # 按配送商分组
        distributor_jobs = defaultdict(list)
        for job in self.jobs:
            distributor_id = getattr(job, 'distributor_id', 0)
            completion_time = schedule_result.job_completion_times.get(job.job_id, 0)
            distributor_jobs[distributor_id].append((job.job_id, completion_time))
        
        total_delivery_time = 0
        on_time_count = 0
        
        # 贪婪批次构造
        for distributor_id, jobs in distributor_jobs.items():
            remaining_jobs = sorted(jobs, key=lambda x: x[1])  # 按完成时间排序
            
            while remaining_jobs:
                # 贪婪选择批次
                current_batch = []
                current_time = 0
                
                # 选择合适的作业组成批次
                for i, (job_id, completion_time) in enumerate(remaining_jobs):
                    if len(current_batch) == 0:
                        current_batch.append((job_id, completion_time))
                        current_time = completion_time
                    elif len(current_batch) < 4:  # 限制批次大小
                        # 评估添加此作业的收益
                        new_time = max(current_time, completion_time)
                        if new_time - current_time <= 5:  # 时间窗口约束
                            current_batch.append((job_id, completion_time))
                            current_time = new_time
                
                # 创建批次
                if current_batch:
                    job_ids = [job_id for job_id, _ in current_batch]
                    dispatch_time = max(comp_time for _, comp_time in current_batch)
                    
                    batches[batch_id] = {
                        'jobs': job_ids,
                        'dispatch_time': dispatch_time,
                        'distributor': distributor_id,
                        'completion_times': dict(current_batch)
                    }
                    
                    total_delivery_time += dispatch_time
                    
                    # 计算按时交付
                    for job_id in job_ids:
                        job = next(j for j in self.jobs if j.job_id == job_id)
                        due_date = getattr(job, 'due_date', 1000)
                        if dispatch_time <= due_date:
                            on_time_count += 1
                    
                    # 移除已分配的作业
                    for job_id, completion_time in current_batch:
                        remaining_jobs.remove((job_id, completion_time))
                    
                    batch_id += 1
        
        on_time_delivery_rate = on_time_count / max(len(self.jobs), 1)
        delivery_cost = total_delivery_time * 0.1
        
        return DispatchResult(
            batches=batches,
            total_delivery_time=total_delivery_time,
            delivery_cost=delivery_cost,
            on_time_delivery_rate=on_time_delivery_rate
        )


class LocalSearchAlgorithm:
    """
    局部搜索算法
    从初始解开始，通过邻域搜索改进解的质量
    """
    
    def __init__(self, scenario: FlexibleJobShopScenario):
        self.scenario = scenario
        self.jobs = scenario.jobs
        self.machines = scenario.machines
        self.distributors = scenario.distributors
    
    def solve(self, max_iterations: int = 100, initial_method: str = "greedy") -> SolutionResult:
        """使用局部搜索算法求解"""
        start_time = time.time()
        
        # 生成初始解
        if initial_method == "greedy":
            greedy_alg = GreedyConstructionAlgorithm(self.scenario)
            current_solution = greedy_alg.solve()
        else:
            priority_alg = PriorityRuleAlgorithm(self.scenario)
            current_solution = priority_alg.solve()
        
        best_solution = copy.deepcopy(current_solution)
        best_objective = current_solution.total_objective
        
        # 局部搜索循环
        for iteration in range(max_iterations):
            # 生成邻域解
            neighbor_solutions = self._generate_neighbors(current_solution)
            
            # 选择最佳邻域解
            improved = False
            for neighbor in neighbor_solutions:
                if neighbor.total_objective < best_objective:
                    best_solution = copy.deepcopy(neighbor)
                    best_objective = neighbor.total_objective
                    current_solution = neighbor
                    improved = True
                    break
            
            # 如果没有改进，尝试其他邻域
            if not improved and iteration % 10 == 0:
                # 随机重启
                current_solution = self._random_perturbation(current_solution)
        
        solve_time = time.time() - start_time
        best_solution.solve_time = solve_time
        best_solution.algorithm_name = f"Local_Search_{initial_method}_{max_iterations}"
        
        if best_solution is not None:
            return best_solution
        else:
            # 返回一个空的 SolutionResult，避免类型错误
            return SolutionResult(
                schedule_result=ScheduleResult(
                    job_schedules={},
                    job_completion_times={},
                    job_tardiness={},
                    makespan=0,
                    total_tardiness=0,
                    machine_utilization={}
                ),
                dispatch_result=DispatchResult(
                    batches={},
                    total_delivery_time=0,
                    delivery_cost=0.0,
                    on_time_delivery_rate=0.0
                ),
                total_objective=float('inf'),
                solve_time=0.0,
                algorithm_name="Genetic_Algorithm_None"
            )
    
    def _generate_neighbors(self, solution: SolutionResult) -> List[SolutionResult]:
        """生成邻域解"""
        neighbors = []
        
        # 邻域1: 交换作业的机器分配
        neighbors.extend(self._machine_swap_neighborhood(solution))
        
        # 邻域2: 调整批次分配
        neighbors.extend(self._batch_reassignment_neighborhood(solution))
        
        # 邻域3: 工序重排序
        neighbors.extend(self._operation_reordering_neighborhood(solution))
        
        return neighbors
    
    def _machine_swap_neighborhood(self, solution: SolutionResult) -> List[SolutionResult]:
        """机器交换邻域"""
        neighbors = []
        job_schedules = solution.schedule_result.job_schedules
        
        for job_id, schedules in job_schedules.items():
            for i, schedule in enumerate(schedules):
                current_machine = schedule['machine']
                operation_idx = schedule['operation']
                
                # 获取可用机器
                job = next(j for j in self.jobs if j.job_id == job_id)
                operation = job.operations[operation_idx]
                eligible_machines = getattr(operation, 'available_machine_ids', [])
                processing_times = getattr(operation, 'processing_times', {})
                
                # 尝试其他机器
                for new_machine in eligible_machines:
                    if new_machine != current_machine and new_machine in processing_times:
                        # 创建新解
                        new_solution = self._create_neighbor_solution(
                            solution, job_id, operation_idx, new_machine
                        )
                        if new_solution:
                            neighbors.append(new_solution)
                
                # 限制邻域大小
                if len(neighbors) >= 5:
                    break
        
        return neighbors
    
    def _batch_reassignment_neighborhood(self, solution: SolutionResult) -> List[SolutionResult]:
        """批次重分配邻域"""
        neighbors = []
        batches = solution.dispatch_result.batches
        
        # 尝试将作业从一个批次移动到另一个批次
        batch_ids = list(batches.keys())
        for i, batch_id1 in enumerate(batch_ids):
            for j, batch_id2 in enumerate(batch_ids):
                if i != j and len(batches[batch_id1]['jobs']) > 1:
                    # 将作业从batch1移动到batch2
                    job_to_move = batches[batch_id1]['jobs'][0]
                    new_solution = self._create_batch_reassignment_solution(
                        solution, job_to_move, batch_id1, batch_id2
                    )
                    if new_solution:
                        neighbors.append(new_solution)
                
                if len(neighbors) >= 3:
                    break
        
        return neighbors
    
    def _operation_reordering_neighborhood(self, solution: SolutionResult) -> List[SolutionResult]:
        """工序重排序邻域"""
        neighbors = []
        
        # 简化实现：随机交换两个不同作业的工序顺序
        job_ids = list(solution.schedule_result.job_schedules.keys())
        
        for _ in range(3):  # 限制尝试次数
            if len(job_ids) >= 2:
                job1, job2 = random.sample(job_ids, 2)
                new_solution = self._create_reordering_solution(solution, job1, job2)
                if new_solution:
                    neighbors.append(new_solution)
        
        return neighbors
    
    def _create_neighbor_solution(self, solution: SolutionResult, job_id: int, 
                                 operation_idx: int, new_machine: int) -> Optional[SolutionResult]:
        """创建机器分配邻域解"""
        try:
            # 复制当前解
            new_schedules = copy.deepcopy(solution.schedule_result.job_schedules)
            
            # 修改机器分配
            job = next(j for j in self.jobs if j.job_id == job_id)
            operation = job.operations[operation_idx]
            processing_times = getattr(operation, 'processing_times', {})
            
            if new_machine in processing_times:
                # 更新处理时间
                new_schedules[job_id][operation_idx]['machine'] = new_machine
                new_schedules[job_id][operation_idx]['processing_time'] = processing_times[new_machine]
                
                # 重新计算时间
                new_solution = self._recalculate_solution(new_schedules, solution.dispatch_result)
                return new_solution
            
        except Exception:
            pass
        
        return None
    
    def _create_batch_reassignment_solution(self, solution: SolutionResult, job_id: int,
                                          from_batch: int, to_batch: int) -> Optional[SolutionResult]:
        """创建批次重分配邻域解"""
        try:
            new_batches = copy.deepcopy(solution.dispatch_result.batches)
            
            # 移动作业
            new_batches[from_batch]['jobs'].remove(job_id)
            new_batches[to_batch]['jobs'].append(job_id)
            
            # 重新计算派遣时间
            schedule_result = solution.schedule_result
            
            # 更新批次派遣时间
            for batch_id in [from_batch, to_batch]:
                if new_batches[batch_id]['jobs']:
                    completion_times = [
                        schedule_result.job_completion_times.get(jid, 0)
                        for jid in new_batches[batch_id]['jobs']
                    ]
                    new_batches[batch_id]['dispatch_time'] = max(completion_times)
            
            # 创建新的派遣结果
            new_dispatch_result = self._create_dispatch_result_from_batches(new_batches)
            
            # 计算新的总目标
            total_objective = (schedule_result.total_tardiness + 
                              new_dispatch_result.total_delivery_time * 0.1)
            
            return SolutionResult(
                schedule_result=schedule_result,
                dispatch_result=new_dispatch_result,
                total_objective=total_objective,
                solve_time=0,
                algorithm_name="Local_Search_Neighbor"
            )
            
        except Exception:
            pass
        
        return None
    
    def _create_reordering_solution(self, solution: SolutionResult, job1_id: int, job2_id: int) -> Optional[SolutionResult]:
        """创建工序重排序邻域解"""
        # 简化实现：交换两个作业的开始时间
        try:
            new_schedules = copy.deepcopy(solution.schedule_result.job_schedules)
            
            job1_schedules = new_schedules[job1_id]
            job2_schedules = new_schedules[job2_id]
            
            if job1_schedules and job2_schedules:
                # 交换第一个工序的开始时间
                temp_start = job1_schedules[0]['start']
                job1_schedules[0]['start'] = job2_schedules[0]['start']
                job2_schedules[0]['start'] = temp_start
                
                # 重新计算解
                new_solution = self._recalculate_solution(new_schedules, solution.dispatch_result)
                return new_solution
        
        except Exception:
            pass
        
        return None
    
    def _recalculate_solution(self, new_schedules: Dict, dispatch_result: DispatchResult) -> SolutionResult:
        """重新计算解的目标值"""
        # 重新计算作业完成时间和延误
        job_completion_times = {}
        job_tardiness = {}
        
        for job_id, schedules in new_schedules.items():
            if schedules:
                # 重新排序工序并计算时间
                schedules.sort(key=lambda x: x['operation'])
                
                for i, schedule in enumerate(schedules):
                    if i > 0:
                        # 确保工序顺序
                        prev_end = schedules[i-1]['end']
                        if schedule['start'] < prev_end:
                            schedule['start'] = prev_end
                    
                    schedule['end'] = schedule['start'] + schedule['processing_time']
                
                completion_time = schedules[-1]['end']
                job_completion_times[job_id] = completion_time
                
                job = next(j for j in self.jobs if j.job_id == job_id)
                due_date = getattr(job, 'due_date', 1000)
                job_tardiness[job_id] = max(0, completion_time - due_date)
        
        makespan = max(job_completion_times.values()) if job_completion_times else 0
        total_tardiness = sum(job_tardiness.values())
        
        # 计算机器利用率
        machine_utilization = {}
        for machine in self.machines:
            busy_time = sum(
                schedule['processing_time'] 
                for schedules in new_schedules.values()
                for schedule in schedules
                if schedule['machine'] == machine.machine_id
            )
            machine_utilization[machine.machine_id] = busy_time / max(makespan, 1)
        
        new_schedule_result = ScheduleResult(
            job_schedules=new_schedules,
            job_completion_times=job_completion_times,
            job_tardiness=job_tardiness,
            makespan=makespan,
            total_tardiness=total_tardiness,
            machine_utilization=machine_utilization
        )
        
        # 计算总目标
        total_objective = (total_tardiness + dispatch_result.total_delivery_time * 0.1)
        
        return SolutionResult(
            schedule_result=new_schedule_result,
            dispatch_result=dispatch_result,
            total_objective=total_objective,
            solve_time=0,
            algorithm_name="Local_Search_Recalculated"
        )
    
    def _create_dispatch_result_from_batches(self, batches: Dict) -> DispatchResult:
        """从批次信息创建派遣结果"""
        total_delivery_time = sum(batch['dispatch_time'] for batch in batches.values())
        delivery_cost = total_delivery_time * 0.1
        
        # 计算按时交付率
        on_time_count = 0
        total_jobs = sum(len(batch['jobs']) for batch in batches.values())
        
        for batch in batches.values():
            for job_id in batch['jobs']:
                job = next(j for j in self.jobs if j.job_id == job_id)
                due_date = getattr(job, 'due_date', 1000)
                if batch['dispatch_time'] <= due_date:
                    on_time_count += 1
        
        on_time_delivery_rate = on_time_count / max(total_jobs, 1)
        
        return DispatchResult(
            batches=batches,
            total_delivery_time=total_delivery_time,
            delivery_cost=delivery_cost,
            on_time_delivery_rate=on_time_delivery_rate
        )
    
    def _random_perturbation(self, solution: SolutionResult) -> SolutionResult:
        """随机扰动当前解"""
        # 简单的随机扰动：随机改变几个作业的机器分配
        new_schedules = copy.deepcopy(solution.schedule_result.job_schedules)
        
        job_ids = list(new_schedules.keys())
        num_perturbations = min(3, len(job_ids))
        
        for _ in range(num_perturbations):
            if job_ids:
                job_id = random.choice(job_ids)
                schedules = new_schedules[job_id]
                
                if schedules:
                    schedule_idx = random.randint(0, len(schedules) - 1)
                    operation_idx = schedules[schedule_idx]['operation']
                    
                    # 获取可用机器
                    job = next(j for j in self.jobs if j.job_id == job_id)
                    operation = job.operations[operation_idx]
                    eligible_machines = getattr(operation, 'available_machine_ids', [])
                    processing_times = getattr(operation, 'processing_times', {})
                    
                    if len(eligible_machines) > 1:
                        new_machine = random.choice(eligible_machines)
                        if new_machine in processing_times:
                            schedules[schedule_idx]['machine'] = new_machine
                            schedules[schedule_idx]['processing_time'] = processing_times[new_machine]
        
        return self._recalculate_solution(new_schedules, solution.dispatch_result)


class GeneticAlgorithm:
    """
    遗传算法
    通过模拟生物进化过程优化解决方案
    """
    
    def __init__(self, scenario: FlexibleJobShopScenario):
        self.scenario = scenario
        self.jobs = scenario.jobs
        self.machines = scenario.machines
        self.distributors = scenario.distributors
    
    def solve(self, population_size: int = 50, generations: int = 100, 
              mutation_rate: float = 0.1, crossover_rate: float = 0.8) -> SolutionResult:
        """使用遗传算法求解"""
        start_time = time.time()
        
        # 初始化种群
        population = self._initialize_population(population_size)
        
        best_solution = None
        best_fitness = float('inf')
        
        # 进化循环
        for generation in range(generations):
            # 评估种群
            fitness_scores = []
            for individual in population:
                solution = self._decode_individual(individual)
                fitness = solution.total_objective
                fitness_scores.append(fitness)
                
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = solution
            
            # 选择、交叉、变异
            new_population = []
            
            # 精英保留
            elite_size = max(1, population_size // 10)
            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])[:elite_size]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # 生成新个体
            while len(new_population) < population_size:
                # 选择父母
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # 交叉
                if random.random() < crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # 变异
                if random.random() < mutation_rate:
                    child1 = self._mutate(child1)
                if random.random() < mutation_rate:
                    child2 = self._mutate(child2)
                
                new_population.extend([child1, child2])
            
            # 更新种群
            population = new_population[:population_size]
        
        solve_time = time.time() - start_time
        if best_solution:
            best_solution.solve_time = solve_time
            best_solution.algorithm_name = f"Genetic_Algorithm_{population_size}_{generations}"
            return best_solution
        else:
            # 返回一个空的 SolutionResult，避免类型错误
            return SolutionResult(
                schedule_result=ScheduleResult(
                    job_schedules={},
                    job_completion_times={},
                    job_tardiness={},
                    makespan=0,
                    total_tardiness=0,
                    machine_utilization={}
                ),
                dispatch_result=DispatchResult(
                    batches={},
                    total_delivery_time=0,
                    delivery_cost=0.0,
                    on_time_delivery_rate=0.0
                ),
                total_objective=float('inf'),
                solve_time=solve_time,
                algorithm_name=f"Genetic_Algorithm_{population_size}_{generations}_None"
            )
    
    def _initialize_population(self, population_size: int) -> List[Dict]:
        """初始化种群"""
        population = []
        
        for _ in range(population_size):
            individual = self._create_random_individual()
            population.append(individual)
        
        return population
    
    def _create_random_individual(self) -> Dict:
        """创建随机个体"""
        individual = {
            'machine_assignment': {},  # {(job_id, op_idx): machine_id}
            'operation_order': [],     # [(job_id, op_idx), ...]
            'batch_assignment': {}     # {job_id: batch_id}
        }
        
        # 机器分配基因
        operation_list = []
        for job in self.jobs:
            for op_idx, operation in enumerate(job.operations):
                eligible_machines = getattr(operation, 'available_machine_ids', [])
                if eligible_machines:
                    selected_machine = random.choice(eligible_machines)
                    individual['machine_assignment'][(job.job_id, op_idx)] = selected_machine
                    operation_list.append((job.job_id, op_idx))
        
        # 工序顺序基因
        random.shuffle(operation_list)
        individual['operation_order'] = operation_list
        
        # 批次分配基因
        max_batches = min(8, len(self.jobs))
        for job in self.jobs:
            batch_id = random.randint(0, max_batches - 1)
            individual['batch_assignment'][job.job_id] = batch_id
        
        return individual
    
    def _decode_individual(self, individual: Dict) -> SolutionResult:
        """解码个体为解决方案"""
        # 解码调度部分
        schedule_result = self._decode_schedule(individual)
        
        # 解码派遣部分
        dispatch_result = self._decode_dispatch(individual, schedule_result)
        
        # 计算总目标
        total_objective = (schedule_result.total_tardiness + 
                          dispatch_result.total_delivery_time * 0.1)
        
        return SolutionResult(
            schedule_result=schedule_result,
            dispatch_result=dispatch_result,
            total_objective=total_objective,
            solve_time=0,
            algorithm_name="Genetic_Individual"
        )
    
    def _decode_schedule(self, individual: Dict) -> ScheduleResult:
        """解码调度基因"""
        machine_assignment = individual['machine_assignment']
        operation_order = individual['operation_order']
        
        job_schedules = {job.job_id: [] for job in self.jobs}
        job_completion_times = {}
        machine_busy_until = {m.machine_id: 0 for m in self.machines}
        job_last_completion = {job.job_id: getattr(job, 'arrival_time', 0) for job in self.jobs}
        
        # 按照基因中的顺序调度工序
        for job_id, op_idx in operation_order:
            machine_id = machine_assignment.get((job_id, op_idx))
            if machine_id is None:
                continue
            
            job = next(j for j in self.jobs if j.job_id == job_id)
            operation = job.operations[op_idx]
            processing_times = getattr(operation, 'processing_times', {})
            
            if machine_id in processing_times:
                start_time = max(job_last_completion[job_id], machine_busy_until[machine_id])
                processing_time = processing_times[machine_id]
                end_time = start_time + processing_time
                
                job_schedules[job_id].append({
                    'operation': op_idx,
                    'machine': machine_id,
                    'start': start_time,
                    'end': end_time,
                    'processing_time': processing_time
                })
                
                machine_busy_until[machine_id] = end_time
                job_last_completion[job_id] = end_time
        
        # 计算完成时间和延误
        for job in self.jobs:
            if job_schedules[job.job_id]:
                job_schedules[job.job_id].sort(key=lambda x: x['operation'])
                job_completion_times[job.job_id] = job_schedules[job.job_id][-1]['end']
            else:
                job_completion_times[job.job_id] = 0
        
        job_tardiness = {}
        for job in self.jobs:
            completion_time = job_completion_times[job.job_id]
            due_date = getattr(job, 'due_date', 1000)
            job_tardiness[job.job_id] = max(0, completion_time - due_date)
        
        makespan = max(job_completion_times.values()) if job_completion_times else 0
        total_tardiness = sum(job_tardiness.values())
        
        # 机器利用率
        machine_utilization = {}
        for machine in self.machines:
            busy_time = sum(
                schedule['processing_time'] 
                for schedules in job_schedules.values()
                for schedule in schedules
                if schedule['machine'] == machine.machine_id
            )
            machine_utilization[machine.machine_id] = busy_time / max(makespan, 1)
        
        return ScheduleResult(
            job_schedules=job_schedules,
            job_completion_times=job_completion_times,
            job_tardiness=job_tardiness,
            makespan=makespan,
            total_tardiness=total_tardiness,
            machine_utilization=machine_utilization
        )
    
    def _decode_dispatch(self, individual: Dict, schedule_result: ScheduleResult) -> DispatchResult:
        """解码派遣基因"""
        batch_assignment = individual['batch_assignment']
        
        # 按批次分组作业
        batches = defaultdict(list)
        for job_id, batch_id in batch_assignment.items():
            completion_time = schedule_result.job_completion_times.get(job_id, 0)
            batches[batch_id].append((job_id, completion_time))
        
        # 创建批次字典
        batch_dict = {}
        total_delivery_time = 0
        on_time_count = 0
        
        for batch_id, jobs in batches.items():
            if jobs:  # 非空批次
                job_ids = [job_id for job_id, _ in jobs]
                completion_times = [comp_time for _, comp_time in jobs]
                dispatch_time = max(completion_times)
                
                # 确定配送商
                job = next(j for j in self.jobs if j.job_id == job_ids[0])
                distributor_id = getattr(job, 'distributor_id', 0)
                
                batch_dict[batch_id] = {
                    'jobs': job_ids,
                    'dispatch_time': dispatch_time,
                    'distributor': distributor_id,
                    'completion_times': dict(jobs)
                }
                
                total_delivery_time += dispatch_time
                
                # 计算按时交付
                for job_id in job_ids:
                    job = next(j for j in self.jobs if j.job_id == job_id)
                    due_date = getattr(job, 'due_date', 1000)
                    if dispatch_time <= due_date:
                        on_time_count += 1
        
        on_time_delivery_rate = on_time_count / max(len(self.jobs), 1)
        delivery_cost = total_delivery_time * 0.1
        
        return DispatchResult(
            batches=batch_dict,
            total_delivery_time=total_delivery_time,
            delivery_cost=delivery_cost,
            on_time_delivery_rate=on_time_delivery_rate
        )
    
    def _tournament_selection(self, population: List[Dict], fitness_scores: List[float], 
                             tournament_size: int = 3) -> Dict:
        """锦标赛选择"""
        tournament_indices = random.sample(range(len(population)), 
                                         min(tournament_size, len(population)))
        best_idx = min(tournament_indices, key=lambda i: fitness_scores[i])
        return population[best_idx]
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """交叉操作"""
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        
        # 机器分配交叉
        machine_keys = list(parent1['machine_assignment'].keys())
        if machine_keys:
            crossover_point = random.randint(1, len(machine_keys) - 1)
            selected_keys = machine_keys[:crossover_point]
            
            for key in selected_keys:
                child1['machine_assignment'][key] = parent2['machine_assignment'][key]
                child2['machine_assignment'][key] = parent1['machine_assignment'][key]
        
        # 工序顺序交叉（PMX）
        if len(parent1['operation_order']) > 2:
            child1['operation_order'] = self._pmx_crossover(
                parent1['operation_order'], parent2['operation_order']
            )
            child2['operation_order'] = self._pmx_crossover(
                parent2['operation_order'], parent1['operation_order']
            )
        
        # 批次分配交叉
        job_ids = list(parent1['batch_assignment'].keys())
        if job_ids:
            crossover_point = random.randint(1, len(job_ids) - 1)
            selected_jobs = job_ids[:crossover_point]
            
            for job_id in selected_jobs:
                child1['batch_assignment'][job_id] = parent2['batch_assignment'][job_id]
                child2['batch_assignment'][job_id] = parent1['batch_assignment'][job_id]
        
        # Always return children
        return child1, child2
    
    def _pmx_crossover(self, parent1: List, parent2: List) -> List:
        """部分匹配交叉（PMX）"""
        if len(parent1) != len(parent2) or len(parent1) < 2:
            return parent1.copy()
        
        size = len(parent1)
        start = random.randint(0, size - 2)
        end = random.randint(start + 1, size - 1)
        
        child = [None] * size
        
        # 复制交叉段
        child[start:end+1] = parent1[start:end+1]
        
        # 填充其余位置
        for i in range(size):
            if child[i] is None:
                value = parent2[i]
                while value in child[start:end+1]:
                    # 找到映射
                    idx = parent1.index(value)
                    value = parent2[idx]
                child[i] = value
        
        return child
    
    def _mutate(self, individual: Dict) -> Dict:
        """变异操作"""
        mutated = copy.deepcopy(individual)
        
        # 机器分配变异
        if random.random() < 0.3:
            machine_keys = list(mutated['machine_assignment'].keys())
            if machine_keys:
                key = random.choice(machine_keys)
                job_id, op_idx = key
                job = next(j for j in self.jobs if j.job_id == job_id)
                operation = job.operations[op_idx]
                eligible_machines = getattr(operation, 'available_machine_ids', [])
                if eligible_machines:
                    mutated['machine_assignment'][key] = random.choice(eligible_machines)
        
        # 工序顺序变异（交换）
        if random.random() < 0.3:
            order = mutated['operation_order']
            if len(order) >= 2:
                i, j = random.sample(range(len(order)), 2)
                order[i], order[j] = order[j], order[i]
        
        # 批次分配变异
        if random.random() < 0.3:
            job_ids = list(mutated['batch_assignment'].keys())
            if job_ids:
                job_id = random.choice(job_ids)
                max_batches = max(mutated['batch_assignment'].values()) + 1
                mutated['batch_assignment'][job_id] = random.randint(0, max_batches)
        
        return mutated


class HeuristicSolver:
    """启发式算法求解器主类"""
    
    def __init__(self, scenario: FlexibleJobShopScenario):
        self.scenario = scenario
        self.algorithms = {
            'priority_rule': PriorityRuleAlgorithm(scenario),
            'greedy_construction': GreedyConstructionAlgorithm(scenario),
            'local_search': LocalSearchAlgorithm(scenario),
            'genetic_algorithm': GeneticAlgorithm(scenario)
        }
    
    def solve_all(self) -> Dict[str, SolutionResult]:
        """使用所有算法求解"""
        results = {}
        
        print("=== 使用优先规则算法求解 ===")
        results['priority_rule_edd_spt'] = self.algorithms['priority_rule'].solve("EDD", "SPT")
        results['priority_rule_spt_eft'] = self.algorithms['priority_rule'].solve("SPT", "EFT")
        
        print("=== 使用贪婪构造算法求解 ===")
        results['greedy_construction'] = self.algorithms['greedy_construction'].solve()
        
        print("=== 使用局部搜索算法求解 ===")
        results['local_search_greedy'] = self.algorithms['local_search'].solve(50, "greedy")
        results['local_search_priority'] = self.algorithms['local_search'].solve(50, "priority")
        
        print("=== 使用遗传算法求解 ===")
        results['genetic_algorithm'] = self.algorithms['genetic_algorithm'].solve(30, 50, 0.1, 0.8)
        
        return results
    
    def print_comparison(self, results: Dict[str, SolutionResult]):
        """打印算法比较结果"""
        print("\n" + "="*80)
        print("启发式算法求解结果比较")
        print("="*80)
        
        print(f"{'算法名称':<25} {'目标值':<12} {'延误':<10} {'完工时间':<10} {'求解时间':<10}")
        print("-" * 80)
        
        for name, result in results.items():
            print(f"{name:<25} "
                  f"{result.total_objective:<12.2f} "
                  f"{result.schedule_result.total_tardiness:<10} "
                  f"{result.schedule_result.makespan:<10} "
                  f"{result.solve_time:<10.2f}")
        
        # 找到最佳解
        best_name = min(results.keys(), key=lambda x: results[x].total_objective)
        best_result = results[best_name]
        
        print("-" * 80)
        print(f"最佳算法: {best_name}")
        print(f"最佳目标值: {best_result.total_objective:.2f}")
        print(f"最佳延误: {best_result.schedule_result.total_tardiness}")
        print(f"最佳完工时间: {best_result.schedule_result.makespan}")
        
        return best_name, best_result
    
    def print_detailed_solution(self, result: SolutionResult):
        """打印详细解决方案"""
        print(f"\n=== 详细解决方案: {result.algorithm_name} ===")
        
        # 调度结果
        print("\n--- 作业调度结果 ---")
        for job_id, schedules in result.schedule_result.job_schedules.items():
            if schedules:
                print(f"作业 {job_id}:")
                for schedule in schedules:
                    print(f"  工序{schedule['operation']}: "
                          f"机器{schedule['machine']}, "
                          f"时间[{schedule['start']}, {schedule['end']}], "
                          f"处理时间{schedule['processing_time']}")
                completion_time = result.schedule_result.job_completion_times[job_id]
                tardiness = result.schedule_result.job_tardiness[job_id]
                print(f"  完成时间: {completion_time}, 延误: {tardiness}")
        
        # 派遣结果
        print("\n--- 批次派遣结果 ---")
        for batch_id, batch_info in result.dispatch_result.batches.items():
            print(f"批次 {batch_id}: "
                  f"作业{batch_info['jobs']}, "
                  f"派遣时间{batch_info['dispatch_time']}, "
                  f"配送商{batch_info['distributor']}")
        
        # 性能指标
        print(f"\n--- 性能指标 ---")
        print(f"总目标值: {result.total_objective:.2f}")
        print(f"总延误: {result.schedule_result.total_tardiness}")
        print(f"完工时间: {result.schedule_result.makespan}")
        print(f"总交付时间: {result.dispatch_result.total_delivery_time}")
        print(f"按时交付率: {result.dispatch_result.on_time_delivery_rate:.2%}")
        print(f"求解时间: {result.solve_time:.2f}秒")
        
        # 机器利用率
        print(f"\n--- 机器利用率 ---")
        for machine_id, utilization in result.schedule_result.machine_utilization.items():
            print(f"机器 {machine_id}: {utilization:.2%}")


def main():
    """主函数 - 运行所有启发式算法并比较结果"""
    
    # 配置参数
    config = Config()
    config.num_initial_jobs = 8
    config.num_machines = 4
    config.num_distributors = 2
    config.min_operations = 2
    config.max_operations = 4
    
    print("生成FJSP-DP实例...")
    scenario = FlexibleJobShopScenario(config=config)
    
    print(f"实例信息:")
    print(f"- 作业数量: {len(scenario.jobs)}")
    print(f"- 机器数量: {len(scenario.machines)}")
    print(f"- 配送商数量: {len(scenario.distributors)}")
    
    # 打印作业详细信息
    print(f"\n--- 作业详细信息 ---")
    for job in scenario.jobs:
        due_date = getattr(job, 'due_date', 'N/A')
        distributor_id = getattr(job, 'distributor_id', 'N/A')
        print(f"作业 {job.job_id}: {len(job.operations)}个工序, "
              f"截止时间={due_date}, 配送商={distributor_id}")
        for i, op in enumerate(job.operations):
            available_machines = getattr(op, 'available_machine_ids', [])
            processing_times = getattr(op, 'processing_times', {})
            print(f"  工序{i}: 可用机器{available_machines}, "
                  f"处理时间{processing_times}")
    
    # 创建求解器
    solver = HeuristicSolver(scenario)
    
    print(f"\n开始求解...")
    start_time = time.time()
    
    # 求解所有算法
    results = solver.solve_all()
    
    total_time = time.time() - start_time
    print(f"\n所有算法求解完成，总耗时: {total_time:.2f}秒")
    
    # 比较结果
    best_name, best_result = solver.print_comparison(results)
    
    # 打印最佳解的详细信息
    solver.print_detailed_solution(best_result)
    
    # 算法性能分析
    print(f"\n=== 算法性能分析 ===")
    
    # 按不同指标排序
    by_objective = sorted(results.items(), key=lambda x: x[1].total_objective)
    by_tardiness = sorted(results.items(), key=lambda x: x[1].schedule_result.total_tardiness)
    by_makespan = sorted(results.items(), key=lambda x: x[1].schedule_result.makespan)
    by_time = sorted(results.items(), key=lambda x: x[1].solve_time)
    
    print(f"\n按总目标值排序:")
    for i, (name, result) in enumerate(by_objective[:3]):
        print(f"{i+1}. {name}: {result.total_objective:.2f}")
    
    print(f"\n按延误时间排序:")
    for i, (name, result) in enumerate(by_tardiness[:3]):
        print(f"{i+1}. {name}: {result.schedule_result.total_tardiness}")
    
    print(f"\n按完工时间排序:")
    for i, (name, result) in enumerate(by_makespan[:3]):
        print(f"{i+1}. {name}: {result.schedule_result.makespan}")
    
    print(f"\n按求解时间排序:")
    for i, (name, result) in enumerate(by_time[:3]):
        print(f"{i+1}. {name}: {result.solve_time:.2f}秒")
    
    # 保存结果到文件
    try:
        import pickle
        with open('heuristic_results.pkl', 'wb') as f:
            pickle.dump({
                'scenario_config': config,
                'results': results,
                'best_solution': best_result,
                'total_solve_time': total_time
            }, f)
        print(f"\n结果已保存到 heuristic_results.pkl")
    except Exception as e:
        print(f"保存结果时出错: {e}")
    
    return results, best_result


def test_single_algorithm():
    """测试单个算法的详细运行过程"""
    config = Config()
    config.num_initial_jobs = 4
    config.num_machines = 3
    config.num_distributors = 2
    config.min_operations = 2
    config.max_operations = 3
    
    scenario = FlexibleJobShopScenario(config=config)
    
    print("=== 测试优先规则算法 ===")
    priority_alg = PriorityRuleAlgorithm(scenario)
    result = priority_alg.solve("EDD", "SPT")
    
    print(f"目标值: {result.total_objective:.2f}")
    print(f"延误: {result.schedule_result.total_tardiness}")
    print(f"完工时间: {result.schedule_result.makespan}")
    print(f"求解时间: {result.solve_time:.2f}秒")
    
    # 详细调度信息
    print(f"\n详细调度:")
    for job_id, schedules in result.schedule_result.job_schedules.items():
        print(f"作业 {job_id}:")
        for schedule in schedules:
            print(f"  工序{schedule['operation']}: "
                  f"机器{schedule['machine']}, "
                  f"[{schedule['start']}-{schedule['end']}]")


def benchmark_algorithms():
    """算法基准测试 - 多个实例"""
    print("=== 启发式算法基准测试 ===")
    
    # 测试不同规模的实例
    test_configs = [
        {'jobs': 4, 'machines': 3, 'distributors': 2},
        {'jobs': 6, 'machines': 4, 'distributors': 2},
        {'jobs': 8, 'machines': 5, 'distributors': 3},
        {'jobs': 10, 'machines': 6, 'distributors': 3},
    ]
    
    all_results = {}
    
    for i, test_config in enumerate(test_configs):
        print(f"\n--- 测试实例 {i+1}: {test_config['jobs']}作业, "
              f"{test_config['machines']}机器, {test_config['distributors']}配送商 ---")
        
        config = Config()
        config.num_initial_jobs = test_config['jobs']
        config.num_machines = test_config['machines']
        config.num_distributors = test_config['distributors']
        config.min_operations = 2
        config.max_operations = 3
        
        scenario = FlexibleJobShopScenario(config=config)
        solver = HeuristicSolver(scenario)
        
        # 只测试主要算法以节省时间
        results = {}
        results['priority_edd_spt'] = solver.algorithms['priority_rule'].solve("EDD", "SPT")
        results['greedy'] = solver.algorithms['greedy_construction'].solve()
        results['local_search'] = solver.algorithms['local_search'].solve(30, "greedy")
        results['genetic'] = solver.algorithms['genetic_algorithm'].solve(20, 30, 0.1, 0.8)
        
        all_results[f"instance_{i+1}"] = results
        
        # 打印本实例最佳结果
        best_name = min(results.keys(), key=lambda x: results[x].total_objective)
        best_result = results[best_name]
        print(f"最佳算法: {best_name}, 目标值: {best_result.total_objective:.2f}")
    
    # 汇总分析
    print(f"\n=== 基准测试汇总 ===")
    algorithm_names = ['priority_edd_spt', 'greedy', 'local_search', 'genetic']
    
    for alg_name in algorithm_names:
        objectives = [all_results[inst][alg_name].total_objective 
                     for inst in all_results.keys()]
        times = [all_results[inst][alg_name].solve_time 
                for inst in all_results.keys()]
        
        print(f"\n{alg_name}:")
        print(f"  平均目标值: {np.mean(objectives):.2f}")
        print(f"  目标值标准差: {np.std(objectives):.2f}")
        print(f"  平均求解时间: {np.mean(times):.2f}秒")
        print(f"  最佳实例数: {sum(1 for inst in all_results.keys() if min(all_results[inst].keys(), key=lambda x: all_results[inst][x].total_objective) == alg_name)}")
    
    return all_results


def run_all_heuristics_experiment(config, case, seed, **kwargs):
    """
    统一运行所有启发式算法的实验函数
    供batch_runner调用，返回与DQN等算法一致的格式
    
    Args:
        config: 配置对象
        case: 算例对象
        seed: 随机种子
        **kwargs: 其他参数
        
    Returns:
        Dict: 包含所有算法统计信息的结果字典
    """
    import random
    import numpy as np
    import time
    
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    
    start_time = time.time()
    
    # 创建启发式求解器
    solver = HeuristicSolver(case)
    
    # 运行所有算法
    print(f"🚀 运行所有启发式算法 (seed={seed})...")
    all_results = solver.solve_all()
    
    total_time = time.time() - start_time
    
    # 转换为统一的统计格式
    stats = defaultdict(list)
    additional_metrics = {}
    
    # 收集所有算法的统计信息
    for algo_name, result in all_results.items():
        # 基本统计
        stats['episode_rewards'].append(result.total_objective)
        stats['makespans'].append(result.schedule_result.makespan)
        stats['total_tardiness'].append(result.schedule_result.total_tardiness)
        stats['solve_times'].append(result.solve_time)
        
        # 算法特定的统计
        stats[f'{algo_name}_objective'] = [result.total_objective]
        stats[f'{algo_name}_makespan'] = [result.schedule_result.makespan]
        stats[f'{algo_name}_tardiness'] = [result.schedule_result.total_tardiness]
        stats[f'{algo_name}_solve_time'] = [result.solve_time]
        
        # 额外的性能指标
        additional_metrics[algo_name] = {
            'objective_value': result.total_objective,
            'makespan': result.schedule_result.makespan,
            'total_tardiness': result.schedule_result.total_tardiness,
            'delivery_time': result.dispatch_result.total_delivery_time,
            'on_time_rate': result.dispatch_result.on_time_delivery_rate,
            'solve_time': result.solve_time,
            'machine_utilization': result.schedule_result.machine_utilization
        }
    
    # 找到最佳算法
    best_algo = min(all_results.keys(), key=lambda x: all_results[x].total_objective)
    best_result = all_results[best_algo]
    
    # 返回与DQN等算法一致的格式
    return {
        'stats': stats,
        'env': case,  # 返回原始算例
        'additional_metrics': additional_metrics,
        'best_algorithm': best_algo,
        'best_objective': best_result.total_objective,
        'all_results': all_results  # 包含所有原始结果
    }


def run_heuristic_experiment(config, case, seed, algorithm_name=None, **kwargs):
    """
    运行单个启发式算法的实验函数
    支持指定特定算法运行
    
    Args:
        config: 配置对象
        case: 算例对象
        seed: 随机种子
        algorithm_name: 算法名称，如果为None则运行所有算法
        **kwargs: 算法特定参数
        
    Returns:
        Dict: 包含算法统计信息的结果字典
    """
    import random
    import numpy as np
    import time
    
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    
    start_time = time.time()
    
    # 创建启发式求解器
    solver = HeuristicSolver(case)
    
    if algorithm_name is None:
        # 运行所有算法
        return run_all_heuristics_experiment(config, case, seed, **kwargs)
    
    # 运行特定算法
    if algorithm_name.lower() == 'priority_rule':
        # 优先规则算法需要指定规则
        job_rule = kwargs.get('job_priority_rule', 'EDD')
        machine_rule = kwargs.get('machine_selection_rule', 'SPT')
        result = solver.algorithms['priority_rule'].solve(job_rule, machine_rule)
        algo_key = f'priority_rule_{job_rule}_{machine_rule}'
        
    elif algorithm_name.lower() == 'greedy_construction':
        result = solver.algorithms['greedy_construction'].solve()
        algo_key = 'greedy_construction'
        
    elif algorithm_name.lower() == 'local_search':
        max_iter = kwargs.get('max_iterations', 50)
        initial_method = kwargs.get('initial_method', 'greedy')
        result = solver.algorithms['local_search'].solve(max_iter, initial_method)
        algo_key = f'local_search_{initial_method}_{max_iter}'
        
    elif algorithm_name.lower() == 'genetic_algorithm':
        pop_size = kwargs.get('population_size', 30)
        generations = kwargs.get('generations', 50)
        mutation_rate = kwargs.get('mutation_rate', 0.1)
        crossover_rate = kwargs.get('crossover_rate', 0.8)
        result = solver.algorithms['genetic_algorithm'].solve(pop_size, generations, mutation_rate, crossover_rate)
        algo_key = f'genetic_algorithm_{pop_size}_{generations}'
        
    else:
        raise ValueError(f"不支持的启发式算法: {algorithm_name}")
    
    total_time = time.time() - start_time
    
    # 转换为统一的统计格式
    stats = defaultdict(list)
    additional_metrics = {}
    
    # 收集统计信息
    stats['episode_rewards'].append(result.total_objective)
    stats['makespans'].append(result.schedule_result.makespan)
    stats['total_tardiness'].append(result.schedule_result.total_tardiness)
    stats['solve_times'].append(result.solve_time)
    
    # 算法特定的统计
    stats[f'{algo_key}_objective'] = [result.total_objective]
    stats[f'{algo_key}_makespan'] = [result.schedule_result.makespan]
    stats[f'{algo_key}_tardiness'] = [result.schedule_result.total_tardiness]
    stats[f'{algo_key}_solve_time'] = [result.solve_time]
    
    # 额外的性能指标
    additional_metrics[algo_key] = {
        'objective_value': result.total_objective,
        'makespan': result.schedule_result.makespan,
        'total_tardiness': result.schedule_result.total_tardiness,
        'delivery_time': result.dispatch_result.total_delivery_time,
        'on_time_rate': result.dispatch_result.on_time_delivery_rate,
        'solve_time': result.solve_time,
        'machine_utilization': result.schedule_result.machine_utilization
    }
    
    return {
        'stats': stats,
        'env': case,
        'additional_metrics': additional_metrics,
        'algorithm_name': algo_key,
        'objective_value': result.total_objective
    }


if __name__ == "__main__":
    # 运行主要测试
    print("启动FJSP-DP启发式算法求解器")
    print("="*60)
    
    # 选择运行模式
    mode = input("选择运行模式 (1: 完整测试, 2: 单算法测试, 3: 基准测试): ").strip()
    
    if mode == "1":
        main()
    elif mode == "2":
        test_single_algorithm()
    elif mode == "3":
        benchmark_algorithms()
    else:
        print("默认运行完整测试...")
        main()
    
    print("\n求解完成!")
