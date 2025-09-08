import numpy as np
import random
import math
from typing import Dict, List, Tuple, Any
from ..base_algorithm import BaseAlgorithm

class SimulatedAnnealing(BaseAlgorithm):
    """
    模拟退火元启发式算法
    使用模拟退火算法求解生产调度问题
    """
    
    def __init__(self, initial_temperature=1000.0, cooling_rate=0.95, 
                 max_iterations=1000, min_temperature=1e-3):
        super().__init__("SimulatedAnnealing")
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.min_temperature = min_temperature
    
    def solve(self, production_data: Dict, orders_data: Dict, T_internal: float) -> Dict:
        """
        使用模拟退火算法求解生产调度问题
        """
        jobs_data = production_data["jobs"]
        machines_data = production_data["machines"]
        precedence = production_data.get("precedence", {})
        
        # 生成初始解
        current_solution = self._generate_initial_solution(jobs_data, machines_data)
        current_schedule = self._decode_solution(current_solution, jobs_data, machines_data, precedence)
        
        if current_schedule:
            current_metrics = self.evaluate_solution(current_schedule, production_data, orders_data, T_internal)
            current_cost = current_metrics['total_cost'] + (1000 if not current_metrics['feasible'] else 0)
        else:
            current_cost = float('inf')
        
        best_solution = current_solution.copy()
        best_cost = current_cost
        best_schedule = current_schedule
        
        temperature = self.initial_temperature
        
        # 模拟退火循环
        for iteration in range(self.max_iterations):
            if temperature < self.min_temperature:
                break
            
            # 生成邻域解
            neighbor_solution = self._generate_neighbor(current_solution, jobs_data)
            neighbor_schedule = self._decode_solution(neighbor_solution, jobs_data, machines_data, precedence)
            
            if neighbor_schedule:
                neighbor_metrics = self.evaluate_solution(neighbor_schedule, production_data, orders_data, T_internal)
                neighbor_cost = neighbor_metrics['total_cost'] + (1000 if not neighbor_metrics['feasible'] else 0)
            else:
                neighbor_cost = float('inf')
            
            # 计算成本差
            cost_difference = neighbor_cost - current_cost
            
            # 决定是否接受新解
            if cost_difference < 0 or random.random() < math.exp(-cost_difference / temperature):
                current_solution = neighbor_solution
                current_cost = neighbor_cost
                current_schedule = neighbor_schedule
                
                # 更新最佳解
                if current_cost < best_cost:
                    best_solution = current_solution.copy()
                    best_cost = current_cost
                    best_schedule = current_schedule
            
            # 降温
            temperature *= self.cooling_rate
        
        # 使用最佳解
        if best_schedule:
            metrics = self.evaluate_solution(best_schedule, production_data, orders_data, T_internal)
        else:
            # 如果没有可行解，返回空结果
            metrics = self.evaluate_solution([], production_data, orders_data, T_internal)
            best_schedule = []
        
        result = {
            'schedule': best_schedule,
            'metrics': metrics,
            'algorithm': self.name,
            'T_internal': T_internal
        }
        
        self.results = result
        return result
    
    def _generate_initial_solution(self, jobs_data: Dict, machines_data: List[str]) -> List[Tuple]:
        """生成初始解"""
        solution = []
        for job_id, job_info in jobs_data.items():
            for op_idx, (op_id, machine_times) in enumerate(job_info):
                # 随机选择可用机器
                available_machines = list(machine_times.keys())
                if available_machines:
                    machine = random.choice(available_machines)
                    solution.append((job_id, op_id, machine))
        random.shuffle(solution)  # 随机排序操作
        return solution
    
    def _decode_solution(self, solution: List[Tuple], jobs_data: Dict, 
                        machines_data: List[str], precedence: Dict) -> List[Dict]:
        """解码解为调度方案"""
        schedule = []
        machine_finish_times = {m: 0.0 for m in machines_data}
        op_completion_times = {}
        job_current_op = {job_id: 0 for job_id in jobs_data.keys()}
        
        for job_id, op_id, machine_id in solution:
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
    
    def _generate_neighbor(self, current_solution: List[Tuple], jobs_data: Dict) -> List[Tuple]:
        """生成邻域解"""
        neighbor = current_solution.copy()
        
        # 随机选择邻域操作
        operation = random.choice(['swap', 'insert', 'change_machine'])
        
        if operation == 'swap' and len(neighbor) >= 2:
            # 交换两个操作的位置
            idx1, idx2 = random.sample(range(len(neighbor)), 2)
            neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
            
        elif operation == 'insert' and len(neighbor) >= 2:
            # 插入操作到新位置
            idx = random.randint(0, len(neighbor) - 1)
            element = neighbor.pop(idx)
            new_idx = random.randint(0, len(neighbor))
            neighbor.insert(new_idx, element)
            
        elif operation == 'change_machine' and len(neighbor) >= 1:
            # 改变操作的机器分配
            idx = random.randint(0, len(neighbor) - 1)
            job_id, op_id, old_machine = neighbor[idx]
            
            # 获取可用机器
            op_idx = next(i for i, (op, _) in enumerate(jobs_data[job_id]) if op == op_id)
            available_machines = list(jobs_data[job_id][op_idx][1].keys())
            
            if len(available_machines) > 1:
                # 选择不同的机器
                new_machine = random.choice([m for m in available_machines if m != old_machine])
                neighbor[idx] = (job_id, op_id, new_machine)
        
        return neighbor
