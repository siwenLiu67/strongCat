"""
元启发式算法环境适配器 - 支持多种元启发式算法的通用接口
参考dynamic_priority_rule_environment_adapter的设计，提供统一的元启发式算法框架
支持遗传算法、模拟退火、禁忌搜索、粒子群优化等多种元启发式算法
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from abc import ABC, abstractmethod

# 使用绝对导入，与dynamic_priority_rule_environment_adapter.py保持一致
from environment import WarehouseEnvironment
from case_generator import FlexibleJobShopScenario
from config import Config
from dispatch_heuristic import DispatchHeuristic
from variable_neighborhood_search_heuristic import VariableNeighborhoodSearchAlgorithm


class MetaheuristicAlgorithm(ABC):
    """元启发式算法抽象基类"""
    
    @abstractmethod
    def optimize(self, jobs: List, current_time: float, **kwargs) -> List:
        """优化作业排序
        
        Args:
            jobs: 待优化的作业列表
            current_time: 当前时间
            **kwargs: 算法特定参数
            
        Returns:
            优化后的作业排序列表
        """
        pass
    
    @abstractmethod
    def get_algorithm_name(self) -> str:
        """获取算法名称"""
        pass


class MetaheuristicEnvironmentAdapter:
    """元启发式算法环境适配器"""
    
    def __init__(self, config: Config, scenario: FlexibleJobShopScenario, 
                 algorithm: MetaheuristicAlgorithm, algorithm_params: Optional[Dict] = None):
        """初始化适配器
        
        Args:
            config: 配置对象
            scenario: 算例生成器实例
            algorithm: 元启发式算法实例
            algorithm_params: 算法特定参数
        """
        self.base_env = WarehouseEnvironment(config, scenario)
        self.scenario = scenario
        self.config = config
        self.algorithm = algorithm
        self.algorithm_params = algorithm_params or {}
        
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
    
    def apply_metaheuristic_scheduling(self):
        """应用元启发式算法进行调度"""
        available_jobs = self._get_available_jobs()
        if not available_jobs:
            return
        
        # 应用元启发式算法优化排序
        sorted_jobs = self.algorithm.optimize(available_jobs, self.base_env.t, **self.algorithm_params)
        
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


# 具体元启发式算法实现

class GeneticAlgorithm(MetaheuristicAlgorithm):
    """遗传算法实现"""
    
    def __init__(self, population_size: int = 50, generations: int = 100,
                 crossover_rate: float = 0.8, mutation_rate: float = 0.1,
                 selection_method: str = "tournament", fitness_function: str = "total_weighted_tardiness"):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.selection_method = selection_method
        self.fitness_function = fitness_function
        
        # 参数验证
        if selection_method not in ["tournament", "roulette"]:
            raise ValueError(f"不支持的选择方法: {selection_method}。支持: ['tournament', 'roulette']")
        
        if fitness_function not in ["total_weighted_tardiness", "makespan", "composite"]:
            raise ValueError(f"不支持的适应度函数: {fitness_function}。支持: ['total_weighted_tardiness', 'makespan', 'composite']")
    
    def optimize(self, jobs: List, current_time: float, **kwargs) -> List:
        """遗传算法优化作业排序"""
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
    
    def _initialize_population(self, jobs: List) -> List[List]:
        """初始化种群"""
        population = []
        
        # 添加随机排列
        for _ in range(self.population_size):
            shuffled_jobs = jobs.copy()
            random.shuffle(shuffled_jobs)
            population.append(shuffled_jobs)
        
        return population
    
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
    
    def _select_population(self, population: List[List], fitness_scores: List[float]) -> List[List]:
        """选择操作"""
        if self.selection_method == "tournament":
            return self._tournament_selection(population, fitness_scores)
        elif self.selection_method == "roulette":
            return self._roulette_selection(population, fitness_scores)
        else:
            return self._tournament_selection(population, fitness_scores)
    
    def _tournament_selection(self, population: List[List], fitness_scores: List[float]) -> List[List]:
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
    
    def _roulette_selection(self, population: List[List], fitness_scores: List[float]) -> List[List]:
        """轮盘赌选择"""
        # 将适应度转换为选择概率
        min_fitness = min(fitness_scores)
        adjusted_fitness = [f - min_fitness + 1e-6 for f in fitness_scores]  # 避免负值
        total_fitness = sum(adjusted_fitness)
        probabilities = [f / total_fitness for f in adjusted_fitness]
        
        selected_indices = np.random.choice(len(population), size=len(population), p=probabilities)
        return [population[i] for i in selected_indices]
    
    def _crossover_population(self, population: List[List]) -> List[List]:
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
    
    def _order_crossover(self, parent1: List, parent2: List) -> Tuple[List, List]:
        """顺序交叉"""
        size = len(parent1)
        
        # 选择交叉点
        crossover_point1 = random.randint(0, size - 1)
        crossover_point2 = random.randint(crossover_point1 + 1, size)
        
        # 创建子代
        child1 = [None] * size
        child2 = [None] * size
        
        # 复制交叉段
        child1[crossover_point1:crossover_point2] = parent1[crossover_point1:crossover_point2]
        child2[crossover_point1:crossover_point2] = parent2[crossover_point1:crossover_point2]
        
        # 填充剩余位置
        self._fill_remaining_positions(child1, parent2, crossover_point1, crossover_point2)
        self._fill_remaining_positions(child2, parent1, crossover_point1, crossover_point2)
        
        return child1, child2
    
    def _fill_remaining_positions(self, child: List, parent: List, 
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
            while parent_index < size:
                if parent[parent_index] not in child[start:end]:
                    child[child_index] = parent[parent_index]
                    parent_index += 1
                    break
                parent_index += 1
            
            child_index += 1
    
    def _mutate_population(self, population: List[List]) -> List[List]:
        """变异操作"""
        mutated_population = []
        
        for individual in population:
            if random.random() < self.mutation_rate:
                mutated_individual = self._swap_mutation(individual)
                mutated_population.append(mutated_individual)
            else:
                mutated_population.append(individual)
        
        return mutated_population
    
    def _swap_mutation(self, individual: List) -> List:
        """交换变异"""
        mutated = individual.copy()
        
        if len(mutated) > 1:
            idx1, idx2 = random.sample(range(len(mutated)), 2)
            mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        
        return mutated
    
    def get_algorithm_name(self) -> str:
        return f"GeneticAlgorithm_{self.fitness_function}"


class SimulatedAnnealingAlgorithm(MetaheuristicAlgorithm):
    """模拟退火算法实现"""
    
    def __init__(self, initial_temperature: float = 1000, cooling_rate: float = 0.95,
                 max_iterations: int = 1000, fitness_function: str = "total_weighted_tardiness"):
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.fitness_function = fitness_function
    
    def optimize(self, jobs: List, current_time: float, **kwargs) -> List:
        """模拟退火算法优化"""
        if len(jobs) <= 1:
            return jobs
            
        # 这里可以实现完整的模拟退火算法
        # 为了简化，这里使用随机排序作为示例
        import random
        optimized_jobs = jobs.copy()
        random.shuffle(optimized_jobs)
        return optimized_jobs
    
    def get_algorithm_name(self) -> str:
        return f"SimulatedAnnealing_{self.fitness_function}"


class TabuSearchAlgorithm(MetaheuristicAlgorithm):
    """禁忌搜索算法实现"""
    
    def __init__(self, tabu_tenure: int = 10, max_iterations: int = 1000,
                 fitness_function: str = "total_weighted_tardiness"):
        self.tabu_tenure = tabu_tenure
        self.max_iterations = max_iterations
        self.fitness_function = fitness_function
    
    def optimize(self, jobs: List, current_time: float, **kwargs) -> List:
        """禁忌搜索算法优化"""
        if len(jobs) <= 1:
            return jobs
            
        # 这里可以实现完整的禁忌搜索算法
        # 为了简化，这里使用随机排序作为示例
        import random
        optimized_jobs = jobs.copy()
        random.shuffle(optimized_jobs)
        return optimized_jobs
    
    def get_algorithm_name(self) -> str:
        return f"TabuSearch_{self.fitness_function}"


class ParticleSwarmOptimizationAlgorithm(MetaheuristicAlgorithm):
    """粒子群优化算法实现"""
    
    def __init__(self, swarm_size: int = 30, max_iterations: int = 100,
                 cognitive_weight: float = 2.0, social_weight: float = 2.0,
                 fitness_function: str = "total_weighted_tardiness"):
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.fitness_function = fitness_function
    
    def optimize(self, jobs: List, current_time: float, **kwargs) -> List:
        """粒子群优化算法优化"""
        if len(jobs) <= 1:
            return jobs
            
        # 这里可以实现完整的粒子群优化算法
        # 为了简化，这里使用随机排序作为示例
        import random
        optimized_jobs = jobs.copy()
        random.shuffle(optimized_jobs)
        return optimized_jobs
    
    def get_algorithm_name(self) -> str:
        return f"PSO_{self.fitness_function}"


# 工厂函数

def create_metaheuristic_algorithm(algorithm_type: str, **kwargs) -> MetaheuristicAlgorithm:
    """创建元启发式算法实例
    
    Args:
        algorithm_type: 算法类型 ("genetic", "simulated_annealing", "tabu_search", "pso")
        **kwargs: 算法特定参数
        
    Returns:
        元启发式算法实例
    """
    if algorithm_type == "genetic":
        return GeneticAlgorithm(**kwargs)
    elif algorithm_type == "simulated_annealing":
        return SimulatedAnnealingAlgorithm(**kwargs)
    elif algorithm_type == "tabu_search":
        return TabuSearchAlgorithm(**kwargs)
    elif algorithm_type == "pso":
        return ParticleSwarmOptimizationAlgorithm(**kwargs)
    else:
        raise ValueError(f"不支持的算法类型: {algorithm_type}")


# 兼容性包装器

class MetaheuristicEnvironment:
    """兼容性包装器，保持与原始接口相同的API"""
    
    def __init__(self, scenario: FlexibleJobShopScenario, 
                 algorithm_type: str = "genetic", algorithm_params: Optional[Dict] = None):
        """初始化兼容性环境"""
        config = Config()
        algorithm = create_metaheuristic_algorithm(algorithm_type, **(algorithm_params or {}))
        self.adapter = MetaheuristicEnvironmentAdapter(config, scenario, algorithm, algorithm_params)
        self.scenario = scenario
        self.algorithm_type = algorithm_type
        
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
