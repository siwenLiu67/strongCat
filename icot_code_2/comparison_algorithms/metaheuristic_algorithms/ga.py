"""
遗传算法 (Genetic Algorithm)
"""
import random
import numpy as np
import copy
from ..base_algorithm import BaseAlgorithm

class GA_Agent(BaseAlgorithm):
    """
    使用遗传算法求解 FJSP 的智能体。
    """
    def __init__(self, env, config):
        """
        :param env: FjspEnv, a scheduling environment instance.
        :param config: dict, a dictionary containing algorithm-specific hyperparameters.
        """
        super().__init__(env, config)
        # 从配置中加载超参数
        self.generations = self.config.get('generations', 50)
        self.population_size = self.config.get('population_size', 30)
        self.crossover_rate = self.config.get('crossover_rate', 0.8)
        self.mutation_rate = self.config.get('mutation_rate', 0.1)

    def _generate_initial_population(self):
        """生成初始种群，每个个体都是一个随机的调度序列。"""
        population = []
        for _ in range(self.population_size):
            # 创建一个包含所有工序索引的列表并随机打乱
            chromosome = list(range(self.n_ops))
            random.shuffle(chromosome)
            population.append(chromosome)
        return population

    def _decode_and_evaluate(self, chromosome):
        """
        解码染色体（工序序列）为一个完整的调度解，并评估其目标函数值。
        这里使用一个简单的启发式规则来分配机器：选择最早可以开始该工序的机器。
        """
        machine_release_times = np.zeros(self.n_machines)
        op_completion_times = {}  # (job_id, op_id) -> time
        
        machine_assignment = [-1] * self.n_ops

        for op_idx in chromosome:
            op_info = self.env.operations[op_idx]
            job_id, op_id = op_info['job_id'], op_info['op_id']
            
            # 找到前序工序的完成时间
            precedent_completion_time = op_completion_times.get((job_id, op_id - 1), 0)
            
            best_machine = -1
            min_end_time = float('inf')
            
            # 遍历所有可选机器，找到能最早完成该工序的机器
            for m_idx, proc_time, _, _ in op_info['proc_options']:
                start_time = max(machine_release_times[m_idx], precedent_completion_time)
                end_time = start_time + proc_time
                if end_time < min_end_time:
                    min_end_time = end_time
                    best_machine = m_idx
            
            # 更新状态
            machine_release_times[best_machine] = min_end_time
            op_completion_times[(job_id, op_id)] = min_end_time
            machine_assignment[op_idx] = best_machine

        # 为每个作业随机但唯一地分配一个运输任务
        # 注意：这里假设运输任务的数量大于或等于作业数量
        if self.env.num_transporters < self.n_jobs:
             # 如果运输工具不够，则循环使用
            transport_assignment = [i % self.env.num_transporters for i in range(self.n_jobs)]
        else:
            transport_assignment = random.sample(range(self.env.num_transporters), self.n_jobs)

        solution = {
            "op_sequence": chromosome,
            "machine_assignment": machine_assignment,
            "transport_assignment": transport_assignment
        }
        
        results = self.env.evaluate_solution(solution)
        print('GA op_sequence:', chromosome)
        print('GA machine_assignment:', machine_assignment)
        print('GA transport_assignment:', transport_assignment)
        print('GA makespan:', results.get('makespan'))
        print('GA objective_value:', results.get('objective_value'))
        print('GA job_final_completion_times:', results.get('job_final_completion_times', 'N/A'))
        print('GA machine_release_times:', machine_release_times)
        print('GA op_completion_times:', op_completion_times)
        return results['objective_value'], solution, results

    def _selection(self, population, fitnesses):
        """轮盘赌选择"""
        total_fitness = sum(1 / f for f in fitnesses) # 适应度是目标值的倒数
        selection_probs = [(1 / f) / total_fitness for f in fitnesses]
        
        selected_indices = np.random.choice(
            len(population),
            size=self.population_size,
            p=selection_probs
        )
        
        return [population[i] for i in selected_indices]

    def _crossover(self, parent1, parent2):
        """顺序交叉 (Order Crossover, OX)"""
        if random.random() > self.crossover_rate:
            return parent1, parent2

        size = len(parent1)
        child1, child2 = [-1] * size, [-1] * size
        
        # 随机选择交叉点
        start, end = sorted(random.sample(range(size), 2))
        
        # 复制交叉段到子代
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]
        
        # 填充剩余部分
        p1_remaining = [item for item in parent2 if item not in child1]
        p2_remaining = [item for item in parent1 if item not in child2]
        
        idx1, idx2 = 0, 0
        for i in range(size):
            if child1[i] == -1:
                child1[i] = p1_remaining[idx1]
                idx1 += 1
            if child2[i] == -1:
                child2[i] = p2_remaining[idx2]
                idx2 += 1
                
        return child1, child2

    def _mutation(self, chromosome):
        """交换变异"""
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(len(chromosome)), 2)
            chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
        return chromosome

    def solve(self):
        print(f"Running {self.__class__.__name__}...")
        
        population = self._generate_initial_population()
        best_objective = float('inf')
        best_solution = None
        best_results = {}

        for gen in range(self.generations):
            # 评估种群
            fitnesses = []
            solutions = []
            for chrom in population:
                objective, sol, _ = self._decode_and_evaluate(chrom)
                fitnesses.append(objective)
                solutions.append(sol)

            # 寻找当前最佳解
            min_fitness = min(fitnesses)
            if min_fitness < best_objective:
                best_objective = min_fitness
                best_idx = fitnesses.index(min_fitness)
                best_solution = solutions[best_idx]
                # 重新评估以获取完整的指标
                best_results = self.env.evaluate_solution(best_solution)

            # 选择
            selected_population = self._selection(population, fitnesses)
            
            # 交叉和变异
            next_population = []
            for i in range(0, self.population_size, 2):
                parent1, parent2 = selected_population[i], selected_population[i+1]
                child1, child2 = self._crossover(parent1, parent2)
                next_population.append(self._mutation(child1))
                next_population.append(self._mutation(child2))
            
            population = next_population

            if (gen + 1) % 20 == 0:
                print(f"  GA | Generation {gen + 1}/{self.generations} | Best Objective: {best_objective:.2f}")

        print(f"{self.__class__.__name__} finished.")
        return best_solution, best_results
