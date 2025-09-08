import numpy as np
import random
from typing import Dict, List, Tuple, Any
from ..base_algorithm import BaseAlgorithm

class GeneticAlgorithm(BaseAlgorithm):
    """
    遗传算法元启发式算法
    使用遗传算法求解生产调度问题
    """
    
    def __init__(self, population_size=50, generations=100, crossover_rate=0.8, 
                 mutation_rate=0.1, tournament_size=3):
        super().__init__("GeneticAlgorithm")
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
    
    def solve(self, production_data: Dict, orders_data: Dict, T_internal: float) -> Dict:
        """
        使用遗传算法求解生产调度问题
        """
        jobs_data = production_data["jobs"]
        machines_data = production_data["machines"]
        precedence = production_data.get("precedence", {})
        
        # 初始化种群
        population = self._initialize_population(jobs_data, machines_data)
        
        best_solution = None
        best_fitness = float('inf')
        
        # 进化循环
        for generation in range(self.generations):
            # 评估种群
            fitness_scores = []
            for individual in population:
                schedule = self._decode_individual(individual, jobs_data, machines_data, precedence)
                if schedule:
                    metrics = self.evaluate_solution(schedule, production_data, orders_data, T_internal)
                    fitness = metrics['total_cost'] + (1000 if not metrics['feasible'] else 0)
                    fitness_scores.append(fitness)
                    
                    # 更新最佳解
                    if fitness < best_fitness:
                        best_fitness = fitness
                        best_solution = schedule
                else:
                    fitness_scores.append(float('inf'))
            
            # 选择
            selected = self._selection(population, fitness_scores)
            
            # 交叉
            offspring = self._crossover(selected)
            
            # 变异
            offspring = self._mutation(offspring, jobs_data)
            
            # 替换
            population = offspring
        
        # 使用最佳解
        if best_solution:
            metrics = self.evaluate_solution(best_solution, production_data, orders_data, T_internal)
        else:
            # 如果没有可行解，返回空结果
            metrics = self.evaluate_solution([], production_data, orders_data, T_internal)
            best_solution = []
        
        result = {
            'schedule': best_solution,
            'metrics': metrics,
            'algorithm': self.name,
            'T_internal': T_internal
        }
        
        self.results = result
        return result
    
    def _initialize_population(self, jobs_data: Dict, machines_data: List[str]) -> List[List[Tuple]]:
        """初始化种群"""
        population = []
        for _ in range(self.population_size):
            individual = []
            for job_id, job_info in jobs_data.items():
                for op_idx, (op_id, machine_times) in enumerate(job_info):
                    # 随机选择可用机器
                    available_machines = list(machine_times.keys())
                    if available_machines:
                        machine = random.choice(available_machines)
                        individual.append((job_id, op_id, machine))
            random.shuffle(individual)  # 随机排序操作
            population.append(individual)
        return population
    
    def _decode_individual(self, individual: List[Tuple], jobs_data: Dict, 
                          machines_data: List[str], precedence: Dict) -> List[Dict]:
        """解码个体为调度方案"""
        schedule = []
        current_time = 0.0
        machine_finish_times = {m: 0.0 for m in machines_data}
        op_completion_times = {}
        job_current_op = {job_id: 0 for job_id in jobs_data.keys()}
        
        for job_id, op_id, machine_id in individual:
            # 检查紧前工序约束
            pre_list = precedence.get(job_id, {}).get(op_id, [])
            precedents_met = True
            for pre_op_id in pre_list:
                if (job_id, pre_op_id) not in op_completion_times:
                    precedents_met = False
                    break
            
            if not precedents_met:
                continue
            
            # 检查操作顺序
            op_idx = next(i for i, (op, _) in enumerate(jobs_data[job_id]) if op == op_id)
            if op_idx != job_current_op[job_id]:
                continue
            
            # 计算开始时间
            start_time = max(machine_finish_times[machine_id], 
                           op_completion_times.get((job_id, op_id), 0.0))
            
            # 获取处理时间
            proc_time = float(jobs_data[job_id][op_idx][1][machine_id])
            
            # 更新状态
            end_time = start_time + proc_time
            machine_finish_times[machine_id] = end_time
            op_completion_times[(job_id, op_id)] = end_time
            job_current_op[job_id] += 1
            
            schedule.append({
                'job': job_id,
                'op': op_id,
                'machine': machine_id,
                'start': start_time,
                'end': end_time
            })
        
        return schedule
    
    def _selection(self, population: List[List[Tuple]], fitness_scores: List[float]) -> List[List[Tuple]]:
        """锦标赛选择"""
        selected = []
        for _ in range(len(population)):
            tournament = random.sample(list(zip(population, fitness_scores)), self.tournament_size)
            winner = min(tournament, key=lambda x: x[1])[0]
            selected.append(winner)
        return selected
    
    def _crossover(self, parents: List[List[Tuple]]) -> List[List[Tuple]]:
        """顺序交叉"""
        offspring = []
        for i in range(0, len(parents), 2):
            if i + 1 >= len(parents):
                offspring.append(parents[i])
                continue
            
            parent1, parent2 = parents[i], parents[i + 1]
            
            if random.random() < self.crossover_rate:
                # 选择交叉点
                crossover_point = random.randint(1, len(parent1) - 2)
                
                # 创建子代
                child1 = parent1[:crossover_point] + [gene for gene in parent2 
                                                     if gene not in parent1[:crossover_point]]
                child2 = parent2[:crossover_point] + [gene for gene in parent1 
                                                     if gene not in parent2[:crossover_point]]
                
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1, parent2])
        
        return offspring
    
    def _mutation(self, population: List[List[Tuple]], jobs_data: Dict) -> List[List[Tuple]]:
        """交换变异"""
        mutated_population = []
        for individual in population:
            if random.random() < self.mutation_rate:
                # 选择两个位置进行交换
                if len(individual) >= 2:
                    idx1, idx2 = random.sample(range(len(individual)), 2)
                    individual = individual.copy()
                    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
            mutated_population.append(individual)
        return mutated_population
