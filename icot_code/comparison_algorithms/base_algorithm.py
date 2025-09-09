import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import json
from abc import ABC, abstractmethod

class BaseAlgorithm(ABC):
    """
    对比算法基类，定义统一的接口
    """
    
    def __init__(self, name: str):
        self.name = name
        self.results = {}
    
    @abstractmethod
    def solve(self, production_data: Dict, transport_date: Dict, orders_data: Dict) -> Dict:
        """
        求解生产调度问题
        
        Args:
            production_data: 生产数据，包含jobs、machines、precedence
            orders_data: 订单数据
            T_internal: 内部生产截止时间
            
        Returns:
            包含调度结果和性能指标的结果字典
        """
        pass
    
    def evaluate_solution(self, schedule: List[Dict], production_data: Dict, 
                         orders_data: Dict, T_internal: float) -> Dict:
        """
        评估调度解决方案
        
        Args:
            schedule: 调度方案列表，每个元素为{'job': job_id, 'op': op_id, 
                      'machine': machine_id, 'start': start_time, 'end': end_time}
            production_data: 生产数据
            orders_data: 订单数据
            T_internal: 内部生产截止时间
            
        Returns:
            性能指标字典
        """
        # 计算最大完工时间
        if not schedule:
            return {
                'makespan': float('inf'),
                'total_cost': float('inf'),
                'production_cost': float('inf'),
                'transportation_cost': float('inf'),
                'tardiness': float('inf'),
                'feasible': False
            }
        
        makespan = max(op['end'] for op in schedule)
        
        # 计算生产总成本
        production_cost = makespan * 10  # 假设单位时间成本为10
        
        # 计算运输成本（简化计算）
        transportation_cost = self._calculate_transportation_cost(schedule, production_data, orders_data)
        
        # 计算总成本
        total_cost = production_cost + transportation_cost
        
        # 检查是否满足T_internal约束
        feasible = makespan <= T_internal
        
        # 计算延迟
        tardiness = max(0, makespan - T_internal)
        
        return {
            'makespan': makespan,
            'total_cost': total_cost,
            'production_cost': production_cost,
            'transportation_cost': transportation_cost,
            'tardiness': tardiness,
            'feasible': feasible
        }
    
    def _calculate_transportation_cost(self, schedule: List[Dict], 
                                     production_data: Dict, orders_data: Dict) -> float:
        """
        简化计算运输成本
        """
        # 这里使用简化计算，实际应该调用运输模型
        # 假设每个作业的运输成本为其重量的函数
        total_weight = 0
        job_completion_times = {}
        
        # 计算每个作业的完成时间
        for op in schedule:
            job_id = op['job']
            if job_id not in job_completion_times or op['end'] > job_completion_times[job_id]:
                job_completion_times[job_id] = op['end']
        
        # 计算总重量
        for job_id in job_completion_times:
            # 假设每个作业的重量为1（简化）
            total_weight += 1
        
        # 简化运输成本计算
        return total_weight * 50  # 假设单位重量运输成本为50
    
    def save_results(self, filename: str):
        """保存结果到文件"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def load_results(self, filename: str):
        """从文件加载结果"""
        with open(filename, 'r') as f:
            self.results = json.load(f)
