"""
遗传算法启发式求解器 - 使用遗传算法替代优先规则进行调度决策
支持动态环境下的实时重新优化
集成派发启发式算法，实现完整的加工和派发解决方案
使用环境适配器复用WarehouseEnvironment，参考dynamic_priority_rule_heuristic的方式
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import random
from dataclasses import dataclass

from dynamic_priority_rule_environment_adapter import DynamicPriorityRuleEnvironmentAdapter
from environment import WarehouseEnvironment
from data_structures import Job, Operation, Machine, Distributor
from config import Config
from case_generator import FlexibleJobShopScenario
from dispatch_heuristic import DispatchHeuristic


class GeneticAlgorithmHeuristicSolver:
    """
    遗传算法启发式求解器，支持动态环境下的实时重新优化
    每次决策时使用遗传算法对所有未调度的工件进行优化排序
    集成派发启发式算法，实现完整的加工和派发解决方案
    """
    
    def __init__(self, env: WarehouseEnvironment, config: Config, 
                 population_size: int = 50, generations: int = 100,
                 crossover_rate: float = 0.8, mutation_rate: float = 0.1,
                 selection_method: str = "tournament", fitness_function: str = "total_weighted_tardiness"):
        """
        初始化遗传算法求解器
        
        Args:
            env: 仓库环境实例
            config: 配置对象
            population_size: 种群大小
            generations: 进化代数
            crossover_rate: 交叉概率
            mutation_rate: 变异概率
            selection_method: 选择方法 ("tournament", "roulette")
            fitness_function: 适应度函数 ("total_weighted_tardiness", "makespan", "composite")
        """
        self.config = config
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.selection_method = selection_method
        self.fitness_function = fitness_function
        
        # 使用环境适配器
        self.env_adapter = DynamicPriorityRuleEnvironmentAdapter(config, env.case, "EDD")  # 使用EDD作为基础规则
        self.env_adapter.register_machine_idle_callback(self._on_machine_idle)
        self.env_adapter.register_job_arrival_callback(self._on_job_arrival)

        # 初始化调度信息跟踪
        self.schedule_records = []  # 存储所有调度记录
        self.current_time = 0
        self.dispatch_heuristic = DispatchHeuristic()  # 派发启发式算法
        
        # 遗传算法参数验证
        if selection_method not in ["tournament", "roulette"]:
            raise ValueError(f"不支持的选择方法: {selection_method}。支持: ['tournament', 'roulette']")
        
        if fitness_function not in ["total_weighted_tardiness", "makespan", "composite"]:
            raise ValueError(f"不支持的适应度函数: {fitness_function}。支持: ['total_weighted_tardiness', 'makespan', 'composite']")
    
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
                "algorithm_name": f"GeneticAlgorithm_{self.fitness_function}"
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
        """执行派发操作"""
        # 只处理已完成且未派发的作业
        dispatch_jobs = [job for job in self.env_adapter.completed_jobs 
                        if job.job_id in job_ids and job.status == 'completed']
        
        if dispatch_jobs:
            # 设置配送时间
            BASE_DELIVERY_TIME = 1
            PER_JOB_TIME = 0
            batch_size = len(dispatch_jobs)
            delivery_time = BASE_DELIVERY_TIME + PER_JOB_TIME * batch_size
            
            # 更新作业状态为配送中
            for job in dispatch_jobs:
                job.status = 'dispatching'
                job.dispatch_remaining_time = delivery_time
                job.dispatch_start_time = self.current_time
    
    def _on_machine_idle(self, machine_ids: List[int]):
        """机器空闲时的回调函数"""
        current_time = self.env_adapter.current_time
        self._trigger_genetic_optimization(current_time)

    def _on_job_arrival(self, new_jobs_count: int):
        """作业到达时的回调函数"""
        current_time = self.env_adapter.current_time
        self._trigger_genetic_optimization(current_time)

    def _trigger_genetic_optimization(self, current_time: float):
        """
        触发遗传算法优化
        获取所有可用作业，使用遗传算法优化排序并分配给所有空闲机器
        """
        available_jobs = self._get_available_jobs(current_time)
        if available_jobs:
            # 使用遗传算法优化作业排序
            optimized_order = self._genetic_algorithm_optimization(available_jobs, current_time)
            
            # 将优化后的排序应用到调度
            self._schedule_to_all_idle_machines(optimized_order, current_time)
    
    def _genetic_algorithm_optimization(self, jobs: List[Job], current_time: float) -> List[Job]:
        """
        遗传算法优化作业排序
        
        Args:
            jobs: 待优化的作业列表
            current_time: 当前时间
            
        Returns:
            优化后的作业排序列表
        """
        if len(jobs) <= 1:
            return jobs  # 不需要优化单个作业
            
        # 初始化种群
        population = self._initialize_population(jobs)
        
        # 进化过程
        for generation in range(self.generations):
            # 评估适应度
            fitness_scores = [self._evaluate_fitness(individual, current_time) for individual in population]
            
            # 选择
            selected_population = self._select_population(population, fitness_scores)
            
            # 交叉
            new_population = self._crossover_population(selected_population)
            
            # 变异
            new_population = self._mutate_population(new_population)
            
            # 精英保留
            best_individual = population[np.argmax(fitness_scores)]
            new_population[0] = best_individual
            
            population = new_population
        
        # 返回最佳个体
        fitness_scores = [self._evaluate_fitness(individual, current_time) for individual in population]
        best_individual = population[np.argmax(fitness_scores)]
        
        return best_individual
    
    def _initialize_population(self, jobs: List[Job]) -> List[List[Job]]:
        """初始化种群"""
        population = []
        
        # 添加随机排列
        for _ in range(self.population_size):
            shuffled_jobs = jobs.copy()
            random.shuffle(shuffled_jobs)
            population.append(shuffled_jobs)
        
        return population
    
    def _evaluate_fitness(self, job_sequence: List[Job], current_time: float) -> float:
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
    
    def _estimate_total_weighted_tardiness(self, job_sequence: List[Job], current_time: float) -> float:
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
    
    def _estimate_makespan(self, job_sequence: List[Job], current_time: float) -> float:
        """估计最大完成时间"""
        completion_time = current_time
        
        for job in job_sequence:
            remaining_processing_time = self._estimate_remaining_processing_time(job)
            completion_time += remaining_processing_time
        
        return completion_time
    
    def _estimate_remaining_processing_time(self, job: Job) -> float:
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
    
    def _select_population(self, population: List[List[Job]], fitness_scores: List[float]) -> List[List[Job]]:
        """选择操作"""
        if self.selection_method == "tournament":
            return self._tournament_selection(population, fitness_scores)
        elif self.selection_method == "roulette":
            return self._roulette_selection(population, fitness_scores)
        else:
            return self._tournament_selection(population, fitness_scores)
    
    def _tournament_selection(self, population: List[List[Job]], fitness_scores: List[float]) -> List[List[Job]]:
        """锦标赛选择"""
        selected_population = []
        tournament_size = 3
        
        for _ in range(len(population)):
            # 随机选择 tournament_size 个个体
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            
            # 选择适应度最高的个体
            winner_index = tournament_indices[np.argmax(tournament_fitness)]
            selected_population.append(population[winner_index])
        
        return selected_population
    
    def _roulette_selection(self, population: List[List[Job]], fitness_scores: List[float]) -> List[List[Job]]:
        """轮盘赌选择"""
        # 将适应度转换为选择概率
        min_fitness = min(fitness_scores)
        adjusted_fitness = [f - min_fitness + 1e-6 for f in fitness_scores]  # 避免负值
        total_fitness = sum(adjusted_fitness)
        probabilities = [f / total_fitness for f in adjusted_fitness]
        
        selected_indices = np.random.choice(len(population), size=len(population), p=probabilities)
        return [population[i] for i in selected_indices]
    
    def _crossover_population(self, population: List[List[Job]]) -> List[List[Job]]:
        """交叉操作"""
        new_population = []
        
        for i in range(0, len(population), 2):
            if i + 1 < len(population):
                parent1 = population[i]
                parent2 = population[i + 1]
                
                if random.random() < self.crossover_rate:
                    child1, child2 = self._order_crossover(parent1, parent2)
                    new_population.extend([child1, child2])
                else:
                    new_population.extend([parent1, parent2])
            else:
                new_population.append(population[i])
        
        return new_population
    
    def _order_crossover(self, parent1: List[Job], parent2: List[Job]) -> Tuple[List[Job], List[Job]]:
        """顺序交叉"""
        size = len(parent1)
        
        # 选择交叉点
        crossover_point1 = random.randint(0, size - 1)
        crossover_point2 = random.randint(crossover_point1 + 1, size)
        
        # 创建子代 - 使用占位符对象而不是 None
        child1 = [parent1[0]] * size  # 使用第一个作业作为占位符
        child2 = [parent2[0]] * size
        
        # 复制交叉段
        child1[crossover_point1:crossover_point2] = parent1[crossover_point1:crossover_point2]
        child2[crossover_point1:crossover_point2] = parent2[crossover_point1:crossover_point2]
        
        # 调用填充方法
        self._fill_remaining_positions(child1, parent2, crossover_point1, crossover_point2)
        self._fill_remaining_positions(child2, parent1, crossover_point1, crossover_point2)
        
        return child1, child2

    def _fill_remaining_positions(self, child: List[Job], parent: List[Job], 
                                 start: int, end: int):
        """填充剩余位置"""
        size = len(child)
        parent_index = 0
        child_index = 0
        
        while child_index < size:
            # 如果当前位置在交叉段内，跳过
            if start <= child_index < end:
                child_index = end
                continue
            
            # 找到parent中不在交叉段中的下一个元素
            found = False
            while parent_index < size and not found:
                if parent[parent_index] not in child[start:end]:
                    child[child_index] = parent[parent_index]
                    parent_index += 1
                    found = True
                else:
                    parent_index += 1
            
            # 如果已经遍历完所有parent元素，跳出循环
            if parent_index >= size:
                break
                
            child_index += 1
    
    def _mutate_population(self, population: List[List[Job]]) -> List[List[Job]]:
        """变异操作"""
        mutated_population = []
        
        for individual in population:
            if random.random() < self.mutation_rate:
                mutated_individual = self._swap_mutation(individual)
                mutated_population.append(mutated_individual)
            else:
                mutated_population.append(individual)
        
        return mutated_population
    
    def _swap_mutation(self, individual: List[Job]) -> List[Job]:
        """交换变异"""
        mutated = individual.copy()
        
        if len(mutated) > 1:
            idx1, idx2 = random.sample(range(len(mutated)), 2)
            mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        
        return mutated
    
    def _schedule_to_all_idle_machines(self, sorted_jobs: List[Job], current_time: float):
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
    
    def _get_available_jobs(self, current_time: float) -> List[Job]:
        """获取当前时间可用的作业（已到达且未完成的作业）"""
        available_jobs = []
        
        for job in self.env_adapter.available_jobs:
            # 检查作业是否已完成
            if not self._is_job_completed(job):
                available_jobs.append(job)
        
        return available_jobs
    
    def _is_job_completed(self, job: Job) -> bool:
        """检查作业是否已完成"""
        return job.status == 'completed' or job.status == 'dispatched' or job.current_operation >= len(job.operations)
    
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


def run_genetic_algorithm_heuristic(env_config: Dict, **kwargs) -> Dict[str, Any]:
    """
    运行遗传算法启发式算法的便捷函数
    
    Args:
        env_config: 环境配置字典
        **kwargs: 遗传算法参数
        
    Returns:
        DQN兼容的结果字典
    """
    # 创建环境实例
    env = WarehouseEnvironment(**env_config)
    
    # 获取遗传算法参数
    population_size = kwargs.get('population_size', 50)
    generations = kwargs.get('generations', 100)
    crossover_rate = kwargs.get('crossover_rate', 0.8)
    mutation_rate = kwargs.get('mutation_rate', 0.1)
    selection_method = kwargs.get('selection_method', 'tournament')
    fitness_function = kwargs.get('fitness_function', 'total_weighted_tardiness')
    
    # 创建求解器
    solver = GeneticAlgorithmHeuristicSolver(
        env, env.config, population_size, generations,
        crossover_rate, mutation_rate, selection_method, fitness_function
    )
    
    # 求解并返回结果
    return solver.solve()


def run_genetic_algorithm_experiment(config, case, seed, **kwargs):
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
    
    # 获取遗传算法参数
    population_size = kwargs.get('population_size', 50)
    generations = kwargs.get('generations', 100)
    crossover_rate = kwargs.get('crossover_rate', 0.8)
    mutation_rate = kwargs.get('mutation_rate', 0.1)
    selection_method = kwargs.get('selection_method', 'tournament')
    fitness_function = kwargs.get('fitness_function', 'total_weighted_tardiness')
    
    # 创建求解器
    solver = GeneticAlgorithmHeuristicSolver(
        env, config, population_size, generations,
        crossover_rate, mutation_rate, selection_method, fitness_function
    )
    
    # 求解并获取结果
    result = solver.solve()
    
    # 返回与PPO实验兼容的格式
    return result
