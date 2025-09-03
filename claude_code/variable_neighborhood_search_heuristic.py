"""
变邻域搜索算法启发式求解器 - 使用变邻域搜索算法替代优先规则进行调度决策
支持动态环境下的实时重新优化
集成派发启发式算法，实现完整的加工和派发解决方案
基于metaheuristic_environment_adapter框架实现
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import random

from claude_code.metaheuristic_environment_adapter import MetaheuristicAlgorithm
from claude_code.environment import WarehouseEnvironment
from claude_code.data_structures import Job, Operation, Machine, Distributor
from claude_code.config import Config
from claude_code.case_generator import FlexibleJobShopScenario
from claude_code.dispatch_heuristic import DispatchHeuristic


class VariableNeighborhoodSearchAlgorithm(MetaheuristicAlgorithm):
    """变邻域搜索算法实现"""
    
    def __init__(self, max_iterations: int = 100, max_no_improvement: int = 20,
                 neighborhood_structures: int = 4, local_search_depth: int = 10,
                 perturbation_strength: float = 0.3, fitness_function: str = "total_weighted_tardiness"):
        """
        初始化变邻域搜索算法
        
        Args:
            max_iterations: 最大迭代次数
            max_no_improvement: 最大无改进迭代次数
            neighborhood_structures: 邻域结构数量
            local_search_depth: 局部搜索深度
            perturbation_strength: 扰动强度（0-1）
            fitness_function: 适应度函数 ("total_weighted_tardiness", "makespan", "composite")
        """
        self.max_iterations = max_iterations
        self.max_no_improvement = max_no_improvement
        self.neighborhood_structures = neighborhood_structures
        self.local_search_depth = local_search_depth
        self.perturbation_strength = perturbation_strength
        self.fitness_function = fitness_function
        
        # 参数验证
        if fitness_function not in ["total_weighted_tardiness", "makespan", "composite"]:
            raise ValueError(f"不支持的适应度函数: {fitness_function}。支持: ['total_weighted_tardiness', 'makespan', 'composite']")
        
        if not 0 <= perturbation_strength <= 1:
            raise ValueError("扰动强度必须在0到1之间")
    
    def optimize(self, jobs: List, current_time: float, **kwargs) -> List:
        """变邻域搜索算法优化作业排序"""
        if len(jobs) <= 1:
            return jobs  # 不需要优化单个作业
        
        # 初始化当前解
        current_solution = jobs.copy()
        random.shuffle(current_solution)
        current_fitness = self._evaluate_fitness(current_solution, current_time)
        
        best_solution = current_solution.copy()
        best_fitness = current_fitness
        
        no_improvement_count = 0
        iteration = 0
        
        # VNS主循环
        while iteration < self.max_iterations and no_improvement_count < self.max_no_improvement:
            k = 1  # 从第一个邻域结构开始
            
            while k <= self.neighborhood_structures:
                # 扰动当前解
                perturbed_solution = self._perturb_solution(current_solution, k)
                
                # 局部搜索
                local_best_solution, local_best_fitness = self._local_search(
                    perturbed_solution, current_time, k
                )
                
                # 邻域切换决策
                if local_best_fitness > current_fitness:
                    # 找到更好的解，移动到该解并重置邻域
                    current_solution = local_best_solution
                    current_fitness = local_best_fitness
                    k = 1  # 回到第一个邻域
                    
                    # 更新全局最优解
                    if current_fitness > best_fitness:
                        best_solution = current_solution.copy()
                        best_fitness = current_fitness
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                else:
                    # 切换到下一个邻域
                    k += 1
                    no_improvement_count += 1
            
            iteration += 1
        
        return best_solution
    
    def _perturb_solution(self, solution: List, neighborhood_type: int) -> List:
        """根据邻域类型扰动解"""
        perturbed = solution.copy()
        
        if neighborhood_type == 1:
            # 交换扰动
            return self._swap_perturbation(perturbed)
        elif neighborhood_type == 2:
            # 插入扰动
            return self._insert_perturbation(perturbed)
        elif neighborhood_type == 3:
            # 反转扰动
            return self._reverse_perturbation(perturbed)
        elif neighborhood_type == 4:
            # 块交换扰动
            return self._block_swap_perturbation(perturbed)
        else:
            return perturbed
    
    def _swap_perturbation(self, solution: List) -> List:
        """交换扰动：随机交换两个位置"""
        if len(solution) <= 1:
            return solution
        
        perturbed = solution.copy()
        idx1, idx2 = random.sample(range(len(perturbed)), 2)
        perturbed[idx1], perturbed[idx2] = perturbed[idx2], perturbed[idx1]
        return perturbed
    
    def _insert_perturbation(self, solution: List) -> List:
        """插入扰动：随机选择一个元素插入到随机位置"""
        if len(solution) <= 1:
            return solution
        
        perturbed = solution.copy()
        from_idx = random.randint(0, len(perturbed) - 1)
        to_idx = random.randint(0, len(perturbed) - 1)
        
        if from_idx != to_idx:
            element = perturbed.pop(from_idx)
            perturbed.insert(to_idx, element)
        
        return perturbed
    
    def _reverse_perturbation(self, solution: List) -> List:
        """反转扰动：反转随机子序列"""
        if len(solution) <= 1:
            return solution
        
        perturbed = solution.copy()
        start = random.randint(0, len(perturbed) - 2)
        end = random.randint(start + 1, len(perturbed) - 1)
        
        perturbed[start:end+1] = reversed(perturbed[start:end+1])
        return perturbed
    
    def _block_swap_perturbation(self, solution: List) -> List:
        """块交换扰动：交换两个随机块"""
        if len(solution) <= 3:
            return solution
        
        perturbed = solution.copy()
        
        # 选择第一个块
        start1 = random.randint(0, len(perturbed) - 3)
        size1 = random.randint(1, min(3, len(perturbed) - start1 - 2))
        end1 = start1 + size1
        
        # 选择第二个块（不与第一个块重叠）
        start2_options = [i for i in range(len(perturbed)) if i < start1 or i > end1]
        if not start2_options:
            return perturbed
            
        start2 = random.choice(start2_options)
        size2 = random.randint(1, min(3, len(perturbed) - start2))
        end2 = start2 + size2
        
        # 执行块交换
        block1 = perturbed[start1:end1]
        block2 = perturbed[start2:end2]
        
        # 调整位置以避免索引冲突
        if start1 < start2:
            perturbed[start1:start1+len(block2)] = block2
            perturbed[start2+len(block2)-len(block1):start2+len(block2)] = block1
        else:
            perturbed[start2:start2+len(block1)] = block1
            perturbed[start1+len(block1)-len(block2):start1+len(block1)] = block2
        
        return perturbed
    
    def _local_search(self, initial_solution: List, current_time: float, 
                     neighborhood_type: int) -> Tuple[List, float]:
        """局部搜索：在当前邻域内寻找最优解"""
        current_solution = initial_solution.copy()
        current_fitness = self._evaluate_fitness(current_solution, current_time)
        
        best_solution = current_solution.copy()
        best_fitness = current_fitness
        
        improvement_found = True
        depth = 0
        
        while improvement_found and depth < self.local_search_depth:
            improvement_found = False
            
            # 生成邻域解
            neighborhood = self._generate_neighborhood(current_solution, neighborhood_type)
            
            for neighbor in neighborhood:
                neighbor_fitness = self._evaluate_fitness(neighbor, current_time)
                
                if neighbor_fitness > current_fitness:
                    current_solution = neighbor
                    current_fitness = neighbor_fitness
                    improvement_found = True
                    
                    if current_fitness > best_fitness:
                        best_solution = current_solution.copy()
                        best_fitness = current_fitness
                    
                    break  # 首次改进策略
            
            depth += 1
        
        return best_solution, best_fitness
    
    def _generate_neighborhood(self, solution: List, neighborhood_type: int) -> List[List]:
        """生成指定类型的邻域解"""
        neighborhood = []
        
        if neighborhood_type == 1:
            # 交换邻域：所有可能的交换
            for i in range(len(solution)):
                for j in range(i + 1, len(solution)):
                    neighbor = solution.copy()
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                    neighborhood.append(neighbor)
        
        elif neighborhood_type == 2:
            # 插入邻域：所有可能的插入
            for i in range(len(solution)):
                for j in range(len(solution)):
                    if i != j:
                        neighbor = solution.copy()
                        element = neighbor.pop(i)
                        neighbor.insert(j, element)
                        neighborhood.append(neighbor)
        
        elif neighborhood_type == 3:
            # 反转邻域：所有可能的子序列反转
            for i in range(len(solution)):
                for j in range(i + 1, len(solution)):
                    neighbor = solution.copy()
                    neighbor[i:j+1] = reversed(neighbor[i:j+1])
                    neighborhood.append(neighbor)
        
        elif neighborhood_type == 4:
            # 块交换邻域：所有可能的块交换
            for i in range(len(solution) - 1):
                for j in range(i + 1, len(solution)):
                    for block_size in range(1, min(3, len(solution) - j + 1)):
                        neighbor = solution.copy()
                        # 交换块[i:i+block_size]和[j:j+block_size]
                        block1 = neighbor[i:i+block_size]
                        block2 = neighbor[j:j+block_size]
                        neighbor[i:i+block_size] = block2
                        neighbor[j:j+block_size] = block1
                        neighborhood.append(neighbor)
        
        return neighborhood
    
    def _evaluate_fitness(self, job_sequence: List, current_time: float) -> float:
        """
        评估个体的适应度
        
        Args:
            job_sequence: 作业序列
            current_time: 当前时间
            
        Returns:
            适应度分数（越高越好）
        """
        if self.fitness_function == "total_weighted_tardiness":
            return -self._estimate_total_weighted_tardiness(job_sequence, current_time)
        elif self.fitness_function == "makespan":
            return -self._estimate_makespan(job_sequence, current_time)
        elif self.fitness_function == "composite":
            # 综合目标：最小化总加权延迟时间和最大完成时间
            twt = self._estimate_total_weighted_tardiness(job_sequence, current_time)
            makespan = self._estimate_makespan(job_sequence, current_time)
            return -(0.7 * twt + 0.3 * makespan)
        else:
            return -self._estimate_total_weighted_tardiness(job_sequence, current_time)
    
    def _estimate_total_weighted_tardiness(self, job_sequence: List, current_time: float) -> float:
        """估计总加权延迟时间"""
        total_tardiness = 0
        completion_time = current_time
        
        for job in job_sequence:
            # 估计剩余处理时间
            remaining_processing_time = self._estimate_remaining_processing_time(job)
            completion_time += remaining_processing_time
            
            tardiness = max(0, completion_time - job.due_date)
            weight = getattr(job, 'weight', 1.0)
            total_tardiness += weight * tardiness
        
        return total_tardiness
    
    def _estimate_makespan(self, job_sequence: List, current_time: float) -> float:
        """估计最大完成时间"""
        completion_time = current_time
        
        for job in job_sequence:
            remaining_processing_time = self._estimate_remaining_processing_time(job)
            completion_time += remaining_processing_time
        
        return completion_time
    
    def _estimate_remaining_processing_time(self, job) -> float:
        """估计作业的剩余处理时间"""
        remaining_time = 0
        for i, op in enumerate(job.operations):
            if i >= job.current_operation:
                if op.processing_times:
                    # 取最小处理时间作为估计
                    remaining_time += min(op.processing_times.values())
                else:
                    remaining_time += 1
        return remaining_time
    
    def get_algorithm_name(self) -> str:
        return f"VariableNeighborhoodSearch_{self.fitness_function}"


class VariableNeighborhoodSearchHeuristicSolver:
    """
    变邻域搜索启发式求解器，支持动态环境下的实时重新优化
    每次决策时使用变邻域搜索算法对所有未调度的工件进行优化排序
    集成派发启发式算法，实现完整的加工和派发解决方案
    """
    
    def __init__(self, env: WarehouseEnvironment, config: Config, 
                 max_iterations: int = 100, max_no_improvement: int = 20,
                 neighborhood_structures: int = 4, local_search_depth: int = 10,
                 perturbation_strength: float = 0.3, fitness_function: str = "total_weighted_tardiness"):
        """
        初始化变邻域搜索求解器
        
        Args:
            env: 仓库环境实例
            config: 配置对象
            max_iterations: 最大迭代次数
            max_no_improvement: 最大无改进迭代次数
            neighborhood_structures: 邻域结构数量
            local_search_depth: 局部搜索深度
            perturbation_strength: 扰动强度
            fitness_function: 适应度函数
        """
        self.config = config
        self.max_iterations = max_iterations
        self.max_no_improvement = max_no_improvement
        self.neighborhood_structures = neighborhood_structures
        self.local_search_depth = local_search_depth
        self.perturbation_strength = perturbation_strength
        self.fitness_function = fitness_function
        
        # 使用环境适配器
        from claude_code.dynamic_priority_rule_environment_adapter import DynamicPriorityRuleEnvironmentAdapter
        self.env_adapter = DynamicPriorityRuleEnvironmentAdapter(config, env.case, "EDD")  # 使用EDD作为基础规则
        self.env_adapter.register_machine_idle_callback(self._on_machine_idle)
        self.env_adapter.register_job_arrival_callback(self._on_job_arrival)

        # 初始化调度信息跟踪
        self.schedule_records = []  # 存储所有调度记录
        self.current_time = 0
        self.dispatch_heuristic = DispatchHeuristic()  # 派发启发式算法
        
        # 参数验证
        if fitness_function not in ["total_weighted_tardiness", "makespan", "composite"]:
            raise ValueError(f"不支持的适应度函数: {fitness_function}。支持: ['total_weighted_tardiness', 'makespan', 'composite']")
    
    def _on_machine_idle(self, machine_ids: List[int]):
        """机器空闲时的回调函数"""
        current_time = self.env_adapter.current_time
        self._trigger_vns_optimization(current_time)

    def _on_job_arrival(self, new_jobs_count: int):
        """作业到达时的回调函数"""
        current_time = self.env_adapter.current_time
        self._trigger_vns_optimization(current_time)

    def _trigger_vns_optimization(self, current_time: float):
        """
        触发变邻域搜索优化
        获取所有可用作业，使用VNS算法优化排序并分配给所有空闲机器
        """
        available_jobs = self._get_available_jobs(current_time)
        if available_jobs:
            # 使用变邻域搜索算法优化作业排序
            vns_algorithm = VariableNeighborhoodSearchAlgorithm(
                self.max_iterations, self.max_no_improvement,
                self.neighborhood_structures, self.local_search_depth,
                self.perturbation_strength, self.fitness_function
            )
            optimized_order = vns_algorithm.optimize(available_jobs, current_time)
            
            # 将优化后的排序应用到调度
            self._schedule_to_all_idle_machines(optimized_order, current_time)
    
    def _get_available_jobs(self, current_time: float) -> List:
        """获取当前时间可用的作业（已到达且未完成的作业）"""
        available_jobs = []
        
        for job in self.env_adapter.available_jobs:
            # 检查作业是否已完成
            if not self._is_job_completed(job):
                available_jobs.append(job)
        
        return available_jobs
    
    def _is_job_completed(self, job) -> bool:
        """检查作业是否已完成"""
        return job.status == 'completed' or job.status == 'dispatched' or job.current_operation >= len(job.operations)
    
    def _schedule_to_all_idle_machines(self, sorted_jobs: List, current_time: float):
        """给所有空闲机器分配作业"""
        if not sorted_jobs:
            return
        
        # 获取所有空闲机器
        idle_machines = [machine for machine in self.env_adapter.machines if machine.status == 'waiting']
        
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
                    # 获取处理时间
                    processing_time = operation_to_schedule.processing_times.get(machine.machine_id, 1)
                    
                    # 创建调度动作
                    schedule_action = {'schedule': {job.job_id: machine.machine_id}}
                    
                    # 执行调度动作
                    self.env_adapter.step(schedule_action)
                    
                    # 记录完整的调度信息
                    schedule_record = {
                        'job_id': job.job_id,
                        'operation_index': job.current_operation,
                        'machine_id': machine.machine_id,
                        'start_time': current_time,
                        'end_time': current_time + processing_time,
                        'processing_time': processing_time
                    }
                    self.schedule_records.append(schedule_record)
                    
                    assigned_jobs.add(job.job_id)
                    break
    
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
                "algorithm_name": f"VariableNeighborhoodSearch_{self.fitness_function}"
            }
        }
    
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
            end_time = record.get('end_time', record['time'])
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
            end_time = record.get('end_time', record['time'])
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
                completion_time = max(record.get('end_time', record['time']) for record in job_records)
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
                completion_time = max(record.get('end_time', record['time']) for record in job_records)
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
                completion_time = max(record.get('end_time', record['time']) for record in job_records)
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
                    busy_time += record.get('processing_time', 1)  # 默认处理时间为1
            
            utilization = busy_time / makespan if makespan > 0 else 0
            machine_utilization[machine.machine_id] = utilization
        
        return machine_utilization
    
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
    
    def _dispatch_completed_jobs(self):
        """派发已完成的作业"""
        # 获取已完成的作业
        completed_jobs = self.env_adapter.completed_jobs.copy()
        
        if not completed_jobs:
            return
        
        # 使用派发启发式算法进行派发决策
        state = {
            'completed_jobs': completed_jobs,
            'config': self.config,
            't': self.current_time
        }
        dispatch_action = self.dispatch_heuristic.select_action(state)
        
        # 执行派发动作
        if dispatch_action and 'dispatch' in dispatch_action:
            self.env_adapter.step(dispatch_action)
            
            # 记录派发信息
            for batch_id, job_ids in dispatch_action['dispatch'].items():
                distributor_id = batch_id // 1000  # 从批次ID中提取配送商ID
                for job_id in job_ids:
                    dispatch_record = {
                        'job_id': job_id,
                        'distributor_id': distributor_id,
                        'batch_id': batch_id,
                        'dispatch_time': self.current_time,
                        'type': 'dispatch'
                    }
                    self.schedule_records.append(dispatch_record)
