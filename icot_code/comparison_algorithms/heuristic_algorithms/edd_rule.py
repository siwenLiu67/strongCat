import numpy as np
from typing import Dict, List, Tuple, Any
from ..base_algorithm import BaseAlgorithm

class EDDRule(BaseAlgorithm):
    """
    最早交货期优先 (EDD) 启发式算法
    总是选择交货期最早的工序进行调度
    """
    
    def __init__(self):
        super().__init__("EDD_Rule")
    
    def solve(self, production_data: Dict, orders_data: Dict, T_internal: float) -> Dict:
        """
        使用EDD规则求解生产调度问题
        """
        jobs_data = production_data["jobs"]
        machines_data = production_data["machines"]
        precedence = production_data.get("precedence", {})
        
        # 初始化状态
        current_time = 0.0
        machine_status = {m: {'idle': True, 'job': None, 'op': None, 'finish_time': -1.0} 
                         for m in machines_data}
        job_status = {}
        schedule = []
        completed_ops = {}
        
        # 初始化作业状态
        for job_id, ops in jobs_data.items():
            job_status[job_id] = {
                'current_op_idx': 0,
                'finished': False,
                'release_time': 0.0,
                'due_date': T_internal,  # 简化处理
                'remaining_proc_time': self._calculate_remaining_proc_time(jobs_data, job_id, 0)
            }
        
        # 主调度循环
        while not all(st['finished'] for st in job_status.values()):
            # 推进时间到下一个事件
            current_time = self._advance_time(machine_status, current_time, completed_ops, job_status)
            
            # 获取所有空闲机器
            idle_machines = [m for m, s in machine_status.items() if s['idle']]
            
            if not idle_machines:
                continue
            
            # 为每个空闲机器找到可用的工序
            for machine_id in idle_machines:
                available_ops = self._get_available_ops(jobs_data, precedence, job_status, 
                                                       machine_id, current_time, completed_ops)
                
                if available_ops:
                    # 应用EDD规则：选择交货期最早的工序
                    selected_op = min(available_ops, key=lambda x: job_status[x[0]]['due_date'])
                    job_id, op_id, proc_time = selected_op
                    
                    # 分配工序到机器
                    machine_status[machine_id] = {
                        'idle': False, 
                        'job': job_id, 
                        'op': op_id, 
                        'finish_time': current_time + proc_time
                    }
                    
                    # 更新作业状态
                    job_st = job_status[job_id]
                    job_st['current_op_idx'] += 1
                    if job_st['current_op_idx'] >= len(jobs_data[job_id]):
                        job_st['finished'] = True
                    job_st['remaining_proc_time'] = self._calculate_remaining_proc_time(
                        jobs_data, job_id, job_st['current_op_idx'])
                    
                    # 记录调度
                    schedule.append({
                        'job': job_id,
                        'op': op_id,
                        'machine': machine_id,
                        'start': current_time,
                        'end': current_time + proc_time
                    })
                    
                    # 记录完成的操作
                    completed_ops[(job_id, op_id)] = current_time + proc_time
        
        # 评估解决方案
        metrics = self.evaluate_solution(schedule, production_data, orders_data, T_internal)
        
        result = {
            'schedule': schedule,
            'metrics': metrics,
            'algorithm': self.name,
            'T_internal': T_internal
        }
        
        self.results = result
        return result
    
    def _calculate_remaining_proc_time(self, jobs_data: Dict, job_id: str, op_idx: int) -> float:
        """计算剩余处理时间"""
        remaining_time = 0.0
        ops = jobs_data[job_id]
        for i in range(op_idx, len(ops)):
            _, machine_times = ops[i]
            if machine_times:
                remaining_time += min(machine_times.values())
        return remaining_time
    
    def _advance_time(self, machine_status: Dict, current_time: float, 
                     completed_ops: Dict, job_status: Dict) -> float:
        """推进时间到下一个事件"""
        # 找到最近的完成时间
        finish_times = [s['finish_time'] for s in machine_status.values() if not s['idle']]
        if finish_times:
            next_time = min(finish_times)
        else:
            # 如果没有忙机器，找到下一个释放时间
            release_times = [st['release_time'] for st in job_status.values() 
                           if not st['finished'] and st['release_time'] > current_time]
            if release_times:
                next_time = min(release_times)
            else:
                return current_time  # 没有事件
        
        # 释放完成的机器
        for m_id, ms in machine_status.items():
            if not ms['idle'] and ms['finish_time'] <= next_time + 1e-8:
                completed_ops[(ms['job'], ms['op'])] = ms['finish_time']
                machine_status[m_id] = {'idle': True, 'job': None, 'op': None, 'finish_time': -1.0}
                
                # 更新作业释放时间
                if not job_status[ms['job']]['finished']:
                    job_status[ms['job']]['release_time'] = next_time
        
        return next_time
    
    def _get_available_ops(self, jobs_data: Dict, precedence: Dict, job_status: Dict,
                          machine_id: str, current_time: float, completed_ops: Dict) -> List[Tuple]:
        """获取可用的工序"""
        available_ops = []
        
        for job_id, st in job_status.items():
            if st['finished'] or current_time < st['release_time']:
                continue
            
            op_idx = st['current_op_idx']
            if op_idx >= len(jobs_data[job_id]):
                continue
                
            op_id, machine_times = jobs_data[job_id][op_idx]
            
            # 检查紧前工序约束
            pre_list = precedence.get(job_id, {}).get(op_id, [])
            precedents_met = True
            for pre_op_id in pre_list:
                if (job_id, pre_op_id) not in completed_ops:
                    precedents_met = False
                    break
            
            # 检查机器可用性
            if precedents_met and machine_id in machine_times:
                available_ops.append((job_id, op_id, float(machine_times[machine_id])))
        
        return available_ops
