"""
动态优先规则启发式算法 - 支持动态环境下的实时重新排序
每次决策时都对所有未调度的工件重新进行启发式排序
集成派发启发式算法，实现完整的加工和派发解决方案
使用环境适配器复用WarehouseEnvironment，参考ppo_environment_adaptor的方式
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from dynamic_priority_rule_environment_adapter import DynamicPriorityRuleEnvironmentAdapter
from environment import WarehouseEnvironment
from data_structures import Job, Operation, Machine, Distributor
from config import Config
from case_generator import FlexibleJobShopScenario
from dispatch_heuristic import DispatchHeuristic


class DynamicPriorityRuleHeuristicSolver:
    """
    动态优先规则启发式求解器，支持动态环境下的实时重新排序
    每次决策时都对所有未调度的工件重新进行启发式排序
    集成派发启发式算法，实现完整的加工和派发解决方案
    使用环境适配器复用WarehouseEnvironment
    """
    
    def __init__(self, env: WarehouseEnvironment, config, rule_name):
        """
        初始化求解器
        
        Args:
            env: 仓库环境实例
            rule_name: 优先规则名称，可选值: "EDD", "SPT", "MST", "CR", "GA"
        """
        self.config = config
        self.rule_name = rule_name
        
        # 使用环境适配器而不是直接使用环境
        self.env_adapter = DynamicPriorityRuleEnvironmentAdapter(config, env.case, rule_name)

    
        # 初始化调度信息跟踪
        

        self.dispatch_heuristic = DispatchHeuristic()  # 派发启发式算法
    
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
        machine_utilization = self.env_adapter.calculate_machine_utilization()
        
        # 计算解决时间
        solve_time = time.time() - start_time
        
        # 返回DQN兼容格式的结果
        return {
            "stats": stats,
            "env": self.env_adapter.env,  # 返回实际的环境实例
            "additional_metrics": {
                "solve_time": solve_time,
                "machine_utilization": machine_utilization,
                "algorithm_name": f"DynamicPriorityRule_{self.rule_name}"
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
            _, _, done, _ = self.env_adapter.step()  # 传递一个空的action
                    
            # 更新当前时间
            self.current_time = self.env_adapter.current_time
        
        # 终止检查
        max_time_steps = getattr(self.env_adapter.config, 'max_time_steps', 1000)
        if self.current_time > max_time_steps:
            print(f"⚠️ 警告：达到最大时间步 {max_time_steps}，但仍有作业未完成或未派发")
            print(f"详细状态: 可用作业={len(self.env_adapter.available_jobs)}, 已完成作业={len(self.env_adapter.completed_jobs)}, 已派发作业={len(self.env_adapter.dispatched_jobs)}")
            print(f"剩余动态作业: {self.env_adapter.remaining_dynamic_jobs}")
            
            for job in self.env_adapter.available_jobs + self.env_adapter.completed_jobs:
                if not self._is_job_completed(job):
                    print(f"作业 {job.job_id} 状态: {job.status}, 当前工序: {job.current_operation}/{len(job.operations)}")


    
    def _check_termination(self) -> bool:
        """检查终止条件，参考environment.py中的逻辑"""
        # 计算已到达的总作业数量（初始作业 + 已到达的动态作业）
        total_arrived_jobs = len(self.env_adapter.initial_jobs) + self.env_adapter.dynamic_jobs_arrived
        
        # 终止条件：所有已到达的作业都已派发且没有剩余动态作业
        # 或者达到最大时间步但动态作业没有到达（避免无限等待）
        return (len(self.env_adapter.dispatched_jobs) >= total_arrived_jobs and 
                self.env_adapter.remaining_dynamic_jobs <= 0)
    
    
    
    def _is_job_completed(self, job: Job) -> bool:
        """检查作业是否已完成"""
        # 检查作业状态或是否所有工序都已完成
        if job.status == 'completed' or job.status == 'dispatched':
            return True
        
        # 检查current_operation是否等于工序数量
        if job.current_operation >= len(job.operations):
            return True
        
        return False
    

    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """计算性能指标"""
        makespan = self.current_time
        total_weighted_tardiness = self.env_adapter.total_weighted_tardiness
        tardy_penalty = self.env_adapter.tardy_penalty
        objective_value = total_weighted_tardiness + tardy_penalty
        
        return {
            "makespan": makespan,
            "objective_value": objective_value,
            "tardy_penalty": tardy_penalty,
            "total_weighted_tardiness": total_weighted_tardiness,
            "algorithm_type": self.rule_name
        }


def run_dynamic_priority_rule_experiment(config, case, seed, **kwargs):
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
    
    # 从kwargs获取规则名称，默认为EDD
    rule_name = kwargs.get('rule_name', 'EDD')
    
    # 创建求解器
    solver = DynamicPriorityRuleHeuristicSolver(env, config, rule_name)
    
    # 求解并获取结果
    result = solver.solve()
    
    # 返回与PPO实验兼容的格式
    return result
