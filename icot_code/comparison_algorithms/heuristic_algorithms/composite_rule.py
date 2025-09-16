import numpy as np
from typing import Dict, List, Tuple, Any
from ..base_algorithm import BaseAlgorithm

class CompositeRule(BaseAlgorithm):
    """
    复合调度规则启发式算法
    结合SPT和EDD规则的复合调度策略
    """
    
    def __init__(self, production_data: Dict, orders_data: Dict, transportation_data: Dict):
        super().__init__("Composite_Rule")
        self.production_data = production_data
        self.orders_data = orders_data
        self.transportation_data = transportation_data

        # ---- 可选的显式映射（NEW）----
        self.job_to_market = production_data.get("job_to_market", {})     # job_id -> market
        self.job_to_deadline = production_data.get("job_to_deadline", {}) # job_id -> absolute deadline (float)
    
    def solve(self) -> Dict:
        """
        使用复合规则求解生产调度问题
        """
        jobs_data = self.production_data["jobs"]
        orders_data = self.orders_data
        machines_data = self.production_data["machines"]
        precedence = self.production_data.get("precedence", {})
        production_data = self.production_data
        transport_data = self.transportation_data
        
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
                'due_date': self.job_to_deadline.get(job_id),
                'remaining_proc_time': self._calculate_remaining_proc_time(jobs_data, job_id, 0),
                'urgency': 0.0  # 紧急程度
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
                    # 更新紧急程度
                    self._update_urgency(job_status, current_time)
                    
                    # 应用复合规则：综合考虑处理时间和紧急程度
                    selected_op = self._apply_composite_rule(available_ops, job_status)
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
        metrics = self.evaluate_solution(schedule, production_data, orders_data, transport_data)
        
        result = {
            'schedule': schedule,
            'metrics': metrics,
            'algorithm': self.name
        }
        
        self.results = result
        return result
    
    def _update_urgency(self, job_status: Dict, current_time: float):
        """更新作业的紧急程度"""
        for job_id, st in job_status.items():
            if st['finished']:
                continue
            
            # 紧急程度 = (交货期 - 当前时间) / 剩余处理时间
            time_left = max(st['due_date'] - current_time, 1e-8)
            remaining_time = max(st['remaining_proc_time'], 1e-8)
            st['urgency'] = time_left / remaining_time
    
    def _apply_composite_rule(self, available_ops: List[Tuple], job_status: Dict) -> Tuple:
        """应用复合调度规则"""
        # 计算每个候选工序的得分
        scores = []
        for job_id, op_id, proc_time in available_ops:
            st = job_status[job_id]
            
            # SPT得分（处理时间越短越好）
            spt_score = 1.0 / max(proc_time, 1e-8)
            
            # 紧急程度得分（紧急程度越高越好）
            urgency_score = st['urgency']
            
            # 复合得分 = 0.6 * SPT得分 + 0.4 * 紧急程度得分
            composite_score = 0.6 * spt_score + 0.4 * urgency_score
            
            scores.append((composite_score, job_id, op_id, proc_time))
        
        # 选择得分最高的工序
        return max(scores, key=lambda x: x[0])[1:]
    
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
