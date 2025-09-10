import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import json
from abc import ABC, abstractmethod
from icot_code.models.transportation_model import plan_transportation

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
                         orders_data: Dict) -> Dict:
        """
        评估调度解决方案
        
        Args:
            schedule: 调度方案列表，每个元素为{'job': job_id, 'op': op_id, 
                      'machine': machine_id, 'start': start_time, 'end': end_time}
            production_data: 生产数据
            orders_data: 订单数据
            
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
                'feasible': False
            }
        
        makespan = max(op['end'] for op in schedule)
        
        # 计算生产总成本
        production_cost = makespan * production_data.get('unit_production_cost')
        
        # 计算运输成本
        transportation_cost, production_plan = plan_transportation(schedule, production_data, orders_data)
        
        # 计算总成本
        total_cost = production_cost + transportation_cost
        
        
        feasible = transportation_cost < float('inf')
        
        return {
            'makespan': makespan,
            'total_cost': total_cost,
            'production_cost': production_cost,
            'transportation_cost': transportation_cost,
            'feasible': feasible
        }
    
    
    
    def save_results(self, filename: str):
        """保存结果到文件"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def load_results(self, filename: str):
        """从文件加载结果"""
        with open(filename, 'r') as f:
            self.results = json.load(f)
