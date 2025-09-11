import heapq
from typing import Dict, List, Tuple
from collections import defaultdict

class DispatchHeuristic:
    def __init__(self, base_delivery_time=10, per_job_time=5):
        self.BASE_DELIVERY_TIME = base_delivery_time
        self.PER_JOB_TIME = per_job_time

    def evaluate_batch(self, batch_jobs, current_time):
        """评估一个批次的派送成本和效益"""
        if not batch_jobs:
            return float('inf'), None

        # 估算批次派遣时间：当前时间或最晚完成时间
        dispatch_time = max(current_time, max(j.completed_time for j in batch_jobs))
        batch_size = len(batch_jobs)
        
        # 1. 配送时间成本（这里用一个简单的线性模型）
        delivery_time_cost = self.BASE_DELIVERY_TIME + self.PER_JOB_TIME * batch_size

        # 2. 作业延误成本
        tardiness_cost = sum(max(0, dispatch_time - j.due_date) for j in batch_jobs)

        # 3. 交付短缺惩罚
        shortfall_cost = 0
        # 简化版：这里需要一个更复杂的逻辑来匹配交付需求
        # 假设每个作业都有一个惩罚权重
        shortfall_cost = sum(j.amount * getattr(j, 'penalty_weight', 1) for j in batch_jobs)
        
        # 总成本：我们要最小化它
        total_cost = tardiness_cost + shortfall_cost + delivery_time_cost
        
        return total_cost, dispatch_time

    def select_action(self, state) -> Dict[str, Dict[int, List[int]]]:
        """
        基于成本的派送决策
        """
        completed_jobs = state.get('completed_jobs', [])
        current_time = state.get('t', 0)
        
        unscheduled_jobs = [j for j in completed_jobs if j.status == 'completed']

        if not unscheduled_jobs:
            return {'dispatch': {}}

        best_cost = float('inf')
        best_batch_jobs = None
        
        # 评估所有可能的单个作业批次
        for job in unscheduled_jobs:
            cost, dispatch_time = self.evaluate_batch([job], current_time)
            if cost < best_cost:
                best_cost = cost
                best_batch_jobs = [job]

        # 评估所有可能的双作业批次（可以扩展到更多）
        for i in range(len(unscheduled_jobs)):
            for j in range(i + 1, len(unscheduled_jobs)):
                job1 = unscheduled_jobs[i]
                job2 = unscheduled_jobs[j]
                
                cost, dispatch_time = self.evaluate_batch([job1, job2], current_time)
                if cost < best_cost:
                    best_cost = cost
                    best_batch_jobs = [job1, job2]
        
        if best_batch_jobs:
            # 找到最优批次，准备派送
            batch_id = 1
            batch_jobs_ids = [j.job_id for j in best_batch_jobs]
            print(f"基于成本选择最优批次: 作业 {batch_jobs_ids}, 成本 {best_cost:.2f}")
            return {'dispatch': {batch_id: batch_jobs_ids}}
        else:
            return {'dispatch': {}}