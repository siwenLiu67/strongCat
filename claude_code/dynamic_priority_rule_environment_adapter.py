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
class PriorityRuleScheduler:
    def __init__(self, rule_name: str, env_adapter: Any):
        self.rule_name = rule_name
        self.env_adapter = env_adapter

        self.rule_mapping = {
            "EDD": self._earliest_due_date,
            "SPT": self._shortest_processing_time,
            "MST": self._minimum_slack_time,
            "CR": self._critical_ratio,
        }

        if rule_name not in self.rule_mapping:
            raise ValueError(f"不支持的优先规则: {rule_name}。支持: {list(self.rule_mapping.keys())}")

    def schedule(self, jobs: List, machines: List) -> Dict:
        """
        应用优先规则进行调度，返回动作字典。
        
        Args:
            jobs: 当前可用的作业列表。
            machines: 当前空闲的机器列表。
        """
        if not jobs:
            return {}
        
        # 1. 应用优先规则排序
        rule_func = self.rule_mapping[self.rule_name]
        sorted_jobs = sorted(jobs, key=lambda job: rule_func(job, self.env_adapter.base_env.t))
        
        # 2. 给所有空闲机器分配作业
        return self._schedule_to_all_idle_machines(sorted_jobs, machines)

    def _schedule_to_all_idle_machines(self, sorted_jobs: List, idle_machines: List) -> Dict:
        """
        为所有可用作业找到最优空闲机器进行分配，并返回动作字典。
        """
        if not sorted_jobs:
            return {}
        
        idle_machines_dict = {machine.machine_id: machine for machine in idle_machines}
        if not idle_machines_dict:
            return {}
            
        scheduling_decisions = {}
        
        for job in sorted_jobs:
            if job.current_operation >= len(job.operations):
                continue
            operation_to_schedule = job.operations[job.current_operation]
            
            best_machine_for_job = None
            
            for machine_id in operation_to_schedule.available_machine_ids:
                if machine_id in idle_machines_dict:
                    best_machine_for_job = idle_machines_dict[machine_id]
                    break
            
            if best_machine_for_job:
                scheduling_decisions[job.job_id] = best_machine_for_job.machine_id
                del idle_machines_dict[best_machine_for_job.machine_id]

        if scheduling_decisions:
            return {'schedule': scheduling_decisions}
        else:
            return {}
    
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
    

class DynamicPriorityRuleEnvironmentAdapter:
    """动态优先规则启发式算法的环境适配器"""
    def __init__(self, config: Config, scenario: FlexibleJobShopScenario, rule_name: str):
        self.base_env = WarehouseEnvironment(config, scenario)
        self.scenario = scenario
        self.config = config
        self.rule_name = rule_name
        
        # 1. 核心改动：用一个字典来映射所有调度器类
        self.scheduler_mapping = {
            "EDD": PriorityRuleScheduler,
            "SPT": PriorityRuleScheduler,
            "MST": PriorityRuleScheduler,
            "CR": PriorityRuleScheduler,
            "GA": GeneticAlgorithmScheduler,  # 将遗传算法也作为一个可选项
            "VNS": VariableNeighborhoodSearchScheduler,  # VNS调度器
        }
        
        if rule_name not in self.scheduler_mapping:
            raise ValueError(f"不支持的调度器: {rule_name}。支持: {list(self.scheduler_mapping.keys())}")
            
        # 2. 根据名称动态创建调度器实例
        if rule_name in ["EDD", "SPT", "MST", "CR"]:
            self.scheduler = self.scheduler_mapping[rule_name](rule_name, self)
        elif rule_name == "GA":
            # 为GA调度器传入必要的参数
            self.scheduler = GeneticAlgorithmScheduler(
                population_size=50, 
                generations=30, 
                mutation_rate=0.05, 
                crossover_rate=0.8,
            )
        elif rule_name == "VNS":
            self.scheduler = VariableNeighborhoodSearchScheduler(
            max_iterations=100,  # 假设你在 config 中定义了这个参数
            neighborhood_structures=['swap', 'relocate']
        )


        # 派发启发式算法
        self.dispatch_heuristic = DispatchHeuristic()
        
    
    def reset(self) -> Dict:
        """重置环境"""
        state = self.base_env.reset()
        
        # 重置跟踪状态
        self.last_idle_machines = set(m.machine_id for m in self.base_env.machines if m.status == 'waiting')
        self.last_available_jobs_count = len(self.base_env.available_jobs)
        self.schedule_records = []
        
        return state
    

    def step(self) -> Tuple[Dict, float, bool, Dict]:
        """
        执行一个时间步，根据预设规则自动生成并执行一个动作。
        不再接收外部动作。
        """
        # 2. 根据业务规则自动决策，生成动作
        action = {}
        
        # 检查是否有空闲机器
        needs_scheduling = self.has_idle_machines() and len(self.completed_jobs) < len(self.initial_jobs)+(self.dynamic_jobs_arrived)
        needs_dispatching = len(self.base_env.completed_jobs) > 0
        
        if needs_scheduling:
           # 调用已创建的调度器实例的 schedule 方法
            available_jobs = self._get_available_jobs()
            available_machines = [m for m in self.base_env.machines if m.status == 'waiting']
            
            # 不同的调度器可能需要不同的输入，这里统一传入 jobs 和 machines
            action = self.scheduler.schedule(available_jobs, available_machines)
            print(f"生成调度动作: 有空闲机器或新工件到达。调度动作: {action}")
        elif needs_dispatching:
            # 准备状态信息，供派发启发式算法使用
            state = {
            'completed_jobs': self.base_env.completed_jobs,
            'config': self.base_env.config,
            't': self.base_env.t
             }
            # 假设此方法返回 {'dispatch': {job_id: machine_id, ...}} 或 {}
            action = self.dispatch_heuristic.select_action(state)
            print(f"生成派发动作: 没有需要调度的空闲资源, 派发动作: {action}")    

        # 3. 如果生成的动作字典为空，则将其替换为 'wait' 动作
        # 这确保了 base_env 总是能接收到有效的动作类型
        if not action:
            print("没有可执行的调度或派发，默认执行等待动作。")
            action = {'wait': {}}
            
        # 4. 推进基础环境，执行生成的动作，并更新状态、奖励和完成标志
        state, reward, done, info = self.base_env.step(action)
        
        # 5. 返回更新后的环境信息
        return state, reward, done, info
    

    def has_idle_machines(self) -> bool:
        """
        检查环境中是否有空闲的机器。
        
        Returns:
            bool: 如果至少有一台机器处于'waiting'状态，则返回True。
        """
        # 遍历所有机器，检查其状态
        for machine in self.base_env.machines:
            if machine.status == 'waiting':
                # 找到一个空闲机器，立即返回 True
                return True
                
        # 如果遍历完所有机器都没有找到空闲的，则返回 False
        return False
    
    
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
        return [job for job in self.base_env.available_jobs if not self._is_job_completed(job) and job.status == 'waiting']
    
    
    
    def _schedule_to_all_idle_machines(self, sorted_jobs: List) -> Dict:
        """
        为所有可用作业找到最优空闲机器进行分配，并返回动作字典。
        """
        if not sorted_jobs:
            return {}
        
        idle_machines = {machine.machine_id: machine for machine in self.base_env.machines if machine.status == 'waiting'}
        if not idle_machines:
            return {}
            
        scheduling_decisions = {}
        
        # 遍历已排序的作业，为每个作业找到最合适的空闲机器
        for job in sorted_jobs:
            if job.current_operation >= len(job.operations):
                continue

            operation_to_schedule = job.operations[job.current_operation]
            
            best_machine_for_job = None
            
            # 在所有空闲机器中，为当前作业寻找一个可用的、且最合适的机器
            # 这里需要根据你的具体启发式规则来定义“最合适”
            for machine_id in operation_to_schedule.available_machine_ids:
                if machine_id in idle_machines:
                    # 假设第一个找到的就是最佳的，因为机器可能已按某种规则排序
                    best_machine_for_job = idle_machines[machine_id]
                    break
            
            # 如果找到了合适的空闲机器
            if best_machine_for_job:
                scheduling_decisions[job.job_id] = best_machine_for_job.machine_id
                # 将该机器标记为不再空闲，防止被重复分配
                del idle_machines[best_machine_for_job.machine_id]

        if scheduling_decisions:
            return {'schedule': scheduling_decisions}
        else:
            return {}
    
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


# 兼容性包装器
class DynamicPriorityRuleEnvironment:
    """兼容性包装器，保持与原始接口相同的API"""
    
    def __init__(self, scenario: FlexibleJobShopScenario, rule_name):
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


import random
from typing import List, Dict, Tuple, Any, Optional

class GeneticAlgorithmScheduler:
    def __init__(self, population_size: int, generations: int, mutation_rate: float, crossover_rate: float):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Instance variables to hold jobs and machines for the current scheduling run
        self.jobs = []
        self.machines = []

    def schedule(self, jobs: List[Any], machines: List[Any]) -> Dict[str, Dict[int, int]]:
        """
        Runs the genetic algorithm to find the best schedule.

        Args:
            jobs: The list of jobs to be scheduled.
            machines: The list of available machines.

        Returns:
            A dictionary with the format {'schedule': {job_id: machine_id, ...}}.
        """
        self.jobs = jobs
        self.machines = machines
        
        if not self.jobs:
            return {}

        # 1. Initialize the population of schedules
        population = self._initialize_population()

        best_individual = None
        best_fitness = -1.0

        for generation in range(self.generations):
            # 2. Evaluate the fitness of each schedule in the population
            fitness_scores = [self._calculate_fitness(individual) for individual in population]

            # 3. Find and update the best individual
            current_best_individual = population[fitness_scores.index(max(fitness_scores))]
            current_best_fitness = max(fitness_scores)
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = current_best_individual
                
            # 4. Select parents for the next generation
            selected_parents = self._selection(population, fitness_scores)
            
            # 5. Perform crossover to create a new population
            new_population = []
            for i in range(0, len(selected_parents), 2):
                parent1, parent2 = selected_parents[i], selected_parents[i+1]
                child1, child2 = self._crossover(parent1, parent2)
                new_population.extend([child1, child2])

            # 6. Apply mutation to the new population
            mutated_population = [self._mutation(individual) for individual in new_population]
            
            # Update the population for the next generation
            population = mutated_population

        # 7. Format and return the best solution
        if best_individual:
            return {'schedule': best_individual}
        else:
            return {}


    def _initialize_population(self) -> List[Dict[int, int]]:
        """Randomly generates an initial population of feasible schedules."""
        population = []
        for _ in range(self.population_size):
            individual = {}
            for job in self.jobs:
                valid_machines = [
                    m.machine_id for m in self.machines
                    if m.machine_id in job.operations[job.current_operation].available_machine_ids
                ]
                if valid_machines:
                    individual[job.job_id] = random.choice(valid_machines)
            population.append(individual)
        return population

    def _calculate_fitness(self, individual: Dict[int, int]) -> float:
        """
        Calculates the fitness of a schedule based on its Makespan.
        A lower Makespan results in a higher fitness score.
        """
        if not individual:
            return 0.0

        machine_end_times = {m.machine_id: 0 for m in self.machines}
        
        # Sort jobs by ID for consistent simulation
        scheduled_jobs = sorted(individual.keys())

        for job_id in scheduled_jobs:
            machine_id = individual[job_id]
            job = next(j for j in self.jobs if j.job_id == job_id)
            operation = job.operations[job.current_operation]
            
            processing_time = operation.processing_times.get(machine_id, 0)
            
            machine_end_times[machine_id] += processing_time

        makespan = max(machine_end_times.values()) if machine_end_times else 1
        return 1.0 / makespan

    def _selection(self, population: List, fitness_scores: List[float]) -> List:
        """Selects parents using Roulette Wheel Selection."""
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            return random.choices(population, k=self.population_size)
        
        weights = [score / total_fitness for score in fitness_scores]
        return random.choices(population, weights=weights, k=self.population_size)

    def _crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """Performs one-point crossover."""
        if random.random() > self.crossover_rate:
            return parent1, parent2
            
        jobs_in_parents = list(parent1.keys())
        if not jobs_in_parents or len(jobs_in_parents) < 2:
            return parent1, parent2

        crossover_point = random.randint(1, len(jobs_in_parents) - 1)
        
        child1 = {jobs_in_parents[i]: parent1[jobs_in_parents[i]] for i in range(crossover_point)}
        child1.update({jobs_in_parents[i]: parent2[jobs_in_parents[i]] for i in range(crossover_point, len(jobs_in_parents))})
        
        child2 = {jobs_in_parents[i]: parent2[jobs_in_parents[i]] for i in range(crossover_point)}
        child2.update({jobs_in_parents[i]: parent1[jobs_in_parents[i]] for i in range(crossover_point, len(jobs_in_parents))})
            
        return child1, child2

    def _mutation(self, individual: Dict) -> Dict:
        """Randomly re-assigns one job to a valid machine."""
        if not individual or random.random() > self.mutation_rate:
            return individual
        
        mutated_individual = individual.copy()
        job_to_mutate_id = random.choice(list(mutated_individual.keys()))
        
        # Find the job object corresponding to the ID
        job_to_mutate = next(j for j in self.jobs if j.job_id == job_to_mutate_id)
        
        # Find all valid machines for the current operation
        valid_machines = [
            m.machine_id for m in self.machines
            if m.machine_id in job_to_mutate.operations[job_to_mutate.current_operation].available_machine_ids
        ]
        
        if valid_machines:
            new_machine_id = random.choice(valid_machines)
            mutated_individual[job_to_mutate_id] = new_machine_id
            
        return mutated_individual
    

import random
from typing import List, Dict, Tuple, Any

class VariableNeighborhoodSearchScheduler:
    def __init__(self, max_iterations: int, neighborhood_structures: List[str]):
        """
        初始化变邻域搜索调度器，用于工件排序。
        
        Args:
            max_iterations: 总的最大迭代次数。
            neighborhood_structures: 邻域结构名称列表，例如 ['swap', 'relocate']。
        """
        self.max_iterations = max_iterations
        self.neighborhood_structures = neighborhood_structures
        
        self.jobs = []
        self.machines = []
        
        # 邻域操作的映射，现在操作的是工件序列
        self.neighborhood_mapping = {
            'swap': self._swap_mutation,
            'relocate': self._relocate_mutation,
            'insert': self._insert_mutation,
        }
        

    def _schedule_to_all_idle_machines(self, sorted_jobs: List, idle_machines: List) -> Dict:
        """
        为所有可用作业找到最优空闲机器进行分配，并返回动作字典。
        """
        if not sorted_jobs:
            return {}
        
        idle_machines_dict = {machine.machine_id: machine for machine in idle_machines}
        if not idle_machines_dict:
            return {}
            
        scheduling_decisions = {}
        
        for job in sorted_jobs:
            if job.current_operation >= len(job.operations):
                continue
            operation_to_schedule = job.operations[job.current_operation]
            
            best_machine_for_job = None
            
            for machine_id in operation_to_schedule.available_machine_ids:
                if machine_id in idle_machines_dict:
                    best_machine_for_job = idle_machines_dict[machine_id]
                    break
            
            if best_machine_for_job:
                scheduling_decisions[job.job_id] = best_machine_for_job.machine_id
                del idle_machines_dict[best_machine_for_job.machine_id]

        if scheduling_decisions:
            return {'schedule': scheduling_decisions}
        else:
            return {}


    def schedule(self, jobs: List[Any], machines: List[Any]) -> Dict[str, Dict[int, int]]:
        """
        运行 VNS 算法，找到最优的工件加工顺序。
        
        Args:
            jobs: 当前需要调度的工件列表。
            machines: 可用的机器列表。
        
        Returns:
            List[Any]: 最优的工件对象序列。
        """
        self.jobs = jobs
        self.machines = machines
        
        if not self.jobs:
            return {}
            
        # 1. 初始化一个随机的工件序列
        current_solution = self._get_initial_solution()
        best_solution = current_solution.copy()
        best_makespan = self._calculate_makespan(best_solution)

        k_max = len(self.neighborhood_structures)
        
        for i in range(self.max_iterations):
            k = 0
            while k < k_max:
                # 2. 邻域探索 (Shaking)
                neighborhood_name = self.neighborhood_structures[k]
                neighbor_solution = self._generate_shaking_solution(current_solution, neighborhood_name)
                
                # 3. 局部搜索 (Local Search)
                local_best_solution = self._local_search(neighbor_solution)
                local_best_makespan = self._calculate_makespan(local_best_solution)

                # 4. 比较和更新
                if local_best_makespan < best_makespan:
                    best_solution = local_best_solution.copy()
                    best_makespan = local_best_makespan
                    current_solution = local_best_solution.copy()
                    k = 0  # 找到更好的解，从第一个邻域重新开始搜索
                else:
                    k += 1 # 否则，切换到下一个邻域
                    
        action = self._schedule_to_all_idle_machines(best_solution, self.machines)
        return action

    def _get_initial_solution(self) -> List[Any]:
        """
        生成一个初始工件序列（随机排列）。
        """
        initial_solution = list(self.jobs)
        random.shuffle(initial_solution)
        return initial_solution

    def _calculate_makespan(self, individual: List[Any]) -> float:
        """
        计算给定工件序列的 Makespan。这里包含了**分配决策**。
        分配规则：将工件分配给最早可用的机器。
        """
        if not individual:
            return 0.0

        # Tracks the time when each machine becomes free.
        machine_end_times = {m.machine_id: 0.0 for m in self.machines}
        
        # Simulate the job sequence
        for job in individual:
            operation_to_schedule = job.operations[job.current_operation]
            
            # Find the machine that can finish this operation earliest
            earliest_completion_time = float('inf')
            best_machine_id = None
            
            for machine_id in operation_to_schedule.available_machine_ids:
                if machine_id in machine_end_times:
                    start_time = machine_end_times[machine_id]
                    processing_time = operation_to_schedule.processing_times.get(machine_id, 0)
                    completion_time = start_time + processing_time
                    
                    if completion_time < earliest_completion_time:
                        earliest_completion_time = completion_time
                        best_machine_id = machine_id
            
            # If a suitable machine was found, update its completion time
            if best_machine_id is not None:
                machine_end_times[best_machine_id] = earliest_completion_time

        makespan = max(machine_end_times.values()) if machine_end_times else 1
        return makespan

    def _generate_shaking_solution(self, solution: List[Any], neighborhood_name: str) -> List[Any]:
        """
        在给定的邻域结构中生成一个邻居解（工件序列）。
        """
        if neighborhood_name not in self.neighborhood_mapping:
            return solution
        
        operation = self.neighborhood_mapping[neighborhood_name]
        return operation(solution.copy())

    def _local_search(self, solution: List[Any]) -> List[Any]:
        """
        在所有邻域结构上执行局部搜索，直到无法找到更好的解。
        """
        current_solution = solution.copy()
        current_makespan = self._calculate_makespan(current_solution)
        
        improved = True
        while improved:
            improved = False
            for neighborhood_name in self.neighborhood_structures:
                # 遍历邻域内所有可能的邻居解
                all_neighbors = self._generate_all_neighbors(current_solution, neighborhood_name)
                
                for neighbor_solution in all_neighbors:
                    neighbor_makespan = self._calculate_makespan(neighbor_solution)
                    
                    if neighbor_makespan < current_makespan:
                        current_solution = neighbor_solution
                        current_makespan = neighbor_makespan
                        improved = True
                        break # 找到更好的解，重新开始局部搜索
                if improved:
                    break
                    
        return current_solution
    

    def _generate_all_neighbors(self, solution: List[Any], neighborhood_name: str) -> List[List[Any]]:
        """
        生成一个邻域内所有可能的邻居解。
        """
        neighbors = []
        n = len(solution)

        if neighborhood_name == 'swap':
            if n < 2: return neighbors
            for i in range(n):
                for j in range(i + 1, n):
                    neighbor = solution.copy()
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                    neighbors.append(neighbor)
        
        elif neighborhood_name == 'relocate':
            if n < 2: return neighbors
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    neighbor = solution.copy()
                    job_to_move = neighbor.pop(i)
                    neighbor.insert(j, job_to_move)
                    neighbors.append(neighbor)
        
        elif neighborhood_name == 'insert':
            if n < 2: return neighbors
            for i in range(n):
                for j in range(i + 1, n):
                    # 移动 solution[i] 到 j
                    neighbor = solution.copy()
                    job_to_move = neighbor.pop(i)
                    neighbor.insert(j, job_to_move)
                    neighbors.append(neighbor)
        
        return neighbors

    def _swap_mutation(self, solution: List) -> List:
        """随机交换两个工件的位置。"""
        if len(solution) < 2: return solution
        idx1, idx2 = random.sample(range(len(solution)), 2)
        solution[idx1], solution[idx2] = solution[idx2], solution[idx1]
        return solution
        
    def _relocate_mutation(self, solution: List) -> List:
        """随机将一个工件移动到新的位置。"""
        if len(solution) < 2: return solution
        idx1 = random.choice(range(len(solution)))
        idx2 = random.choice(range(len(solution)))
        while idx1 == idx2:
            idx2 = random.choice(range(len(solution)))
        
        job_to_move = solution.pop(idx1)
        solution.insert(idx2, job_to_move)
        return solution

    def _insert_mutation(self, solution: List) -> List:
        """随机将一个工件插入到新的位置。"""
        return self._relocate_mutation(solution)
