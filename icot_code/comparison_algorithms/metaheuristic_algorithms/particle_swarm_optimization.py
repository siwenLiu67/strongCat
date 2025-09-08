import numpy as np
import random
from typing import Dict, List, Tuple, Any
from ..base_algorithm import BaseAlgorithm

class ParticleSwarmOptimization(BaseAlgorithm):
    """
    粒子群优化元启发式算法
    使用粒子群优化算法求解生产调度问题
    """
    
    def __init__(self, swarm_size=20, max_iterations=50, 
                 cognitive_weight=1.5, social_weight=1.5, inertia_weight=0.7):
        super().__init__("ParticleSwarmOptimization")
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.inertia_weight = inertia_weight
    
    def solve(self, production_data: Dict, orders_data: Dict, T_internal: float) -> Dict:
        """
        使用粒子群优化算法求解生产调度问题
        """
        jobs_data = production_data["jobs"]
        machines_data = production_data["machines"]
        precedence = production_data.get("precedence", {})
        
        # 初始化粒子群
        particles = self._initialize_swarm(jobs_data, machines_data)
        velocities = [np.zeros(len(particles[0])) for _ in range(self.swarm_size)]
        
        personal_best_positions = particles.copy()
        personal_best_costs = [float('inf')] * self.swarm_size
        
        global_best_position = None
        global_best_cost = float('inf')
        global_best_schedule = None
        
        # PSO主循环
        for iteration in range(self.max_iterations):
            for i in range(self.swarm_size):
                # 解码粒子位置为调度方案
                schedule = self._decode_particle(particles[i], jobs_data, machines_data, precedence)
                
                if schedule:
                    metrics = self.evaluate_solution(schedule, production_data, orders_data, T_internal)
                    cost = metrics['total_cost'] + (1000 if not metrics['feasible'] else 0)
                    
                    # 更新个体最优
                    if cost < personal_best_costs[i]:
                        personal_best_costs[i] = cost
                        personal_best_positions[i] = particles[i].copy()
                    
                    # 更新全局最优
                    if cost < global_best_cost:
                        global_best_cost = cost
                        global_best_position = particles[i].copy()
                        global_best_schedule = schedule
            
            # 更新粒子速度和位置
            for i in range(self.swarm_size):
                # 更新速度
                r1, r2 = random.random(), random.random()
                cognitive_component = self.cognitive_weight * r1 * (personal_best_positions[i] - particles[i])
                social_component = self.social_weight * r2 * (global_best_position - particles[i])
                
                velocities[i] = (self.inertia_weight * velocities[i] + 
                                cognitive_component + social_component)
                
                # 更新位置
                particles[i] = particles[i] + velocities[i]
                
                # 确保位置在合理范围内
                particles[i] = np.clip(particles[i], 0, 1)
        
        # 使用最佳解
        if global_best_schedule:
            metrics = self.evaluate_solution(global_best_schedule, production_data, orders_data, T_internal)
        else:
            # 如果没有可行解，返回空结果
            metrics = self.evaluate_solution([], production_data, orders_data, T_internal)
            global_best_schedule = []
        
        result = {
            'schedule': global_best_schedule,
            'metrics': metrics,
            'algorithm': self.name,
            'T_internal': T_internal
        }
        
        self.results = result
        return result
    
    def _initialize_swarm(self, jobs_data: Dict, machines_data: List[str]) -> List[np.ndarray]:
        """初始化粒子群"""
        swarm = []
        total_ops = sum(len(ops) for ops in jobs_data.values())
        
        for _ in range(self.swarm_size):
            # 创建随机位置向量
            position = np.random.random(total_ops * 2)  # 操作顺序 + 机器分配
            
            # 确保操作顺序部分是有序的
            op_order_part = position[:total_ops]
            op_order_part = np.argsort(op_order_part) / (total_ops - 1)  # 归一化到[0,1]
            
            position[:total_ops] = op_order_part
            swarm.append(position)
        
        return swarm
    
    def _decode_particle(self, position: np.ndarray, jobs_data: Dict, 
                        machines_data: List[str], precedence: Dict) -> List[Dict]:
        """解码粒子位置为调度方案"""
        total_ops = sum(len(ops) for ops in jobs_data.values())
        op_order_part = position[:total_ops]
        machine_part = position[total_ops:]
        
        # 根据位置向量确定操作顺序
        op_indices = np.argsort(op_order_part)
        
        # 创建操作序列
        solution = []
        op_counter = 0
        
        for job_id, job_ops in jobs_data.items():
            for op_idx, (op_id, machine_times) in enumerate(job_ops):
                # 确定机器分配
                machine_prob = machine_part[op_counter]
                available_machines = list(machine_times.keys())
                
                if available_machines:
                    # 根据概率选择机器
                    machine_idx = int(machine_prob * len(available_machines)) % len(available_machines)
                    machine_id = available_machines[machine_idx]
                    solution.append((job_id, op_id, machine_id))
                
                op_counter += 1
        
        # 根据操作顺序排序
        sorted_solution = [solution[i] for i in op_indices if i < len(solution)]
        
        # 解码为调度方案
        schedule = []
        machine_finish_times = {m: 0.0 for m in machines_data}
        op_completion_times = {}
        job_current_op = {job_id: 0 for job_id in jobs_data.keys()}
        
        for job_id, op_id, machine_id in sorted_solution:
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
