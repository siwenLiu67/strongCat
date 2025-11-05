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

    def select_action_(self, state, method='cost_based') -> Dict[str, Dict[int, List[int]]]:
        """
        派送决策主方法
        
        Args:
            state: 环境状态
            method: 选择方法，'cost_based' 或 'greedy'
        """
        completed_jobs = state.get('completed_jobs', [])
        current_time = state.get('t', 0)
        
        unscheduled_jobs = [j for j in completed_jobs if j.status == 'completed']

        if not unscheduled_jobs:
            return {'dispatch': {}}

        
        return self.select_action(state) if method == 'greedy' else self._select_action_cost_based(unscheduled_jobs, current_time)
        

    def _select_action_cost_based(self, unscheduled_jobs, current_time):
        """基于成本的派送决策"""
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

    def select_action(self, state):
        """基于贪心算法的派送决策"""
        unscheduled_jobs = [j for j in state['completed_jobs'] if j.status == 'completed']
        current_time = state['current_time']
        batches = self.greedy_batch_selection(unscheduled_jobs, current_time)
        
        if batches:
            print(f"基于贪心算法选择批次: {batches}")
            return {'dispatch': batches}
        else:
            return {'dispatch': {}}
        
    def group_by_distributor(self, jobs):
        """按配送商分组作业"""
        jobs_by_distributor = defaultdict(list)
        for job in jobs:
            jobs_by_distributor[job.distributor_id].append(job)
        return jobs_by_distributor

    def calculate_priority_score(self, job, current_time):
        """计算作业的优先级评分
        
        评分越高表示优先级越高，应该优先配送
        """
        # 1. 紧急作业优先级最高
        if getattr(job, 'is_urgent', False):
            urgency_score = 1000
        else:
            urgency_score = 0
        
        # 2. 基于截止时间的紧迫性
        time_remaining = max(0, job.due_date - current_time)
        if time_remaining == 0:
            due_date_score = 100  # 已经超期的作业
        else:
            due_date_score = 100 / (time_remaining + 1)  # 时间越少，分数越高
        
        # 3. 基于作业数量的权重
        amount_score = job.amount * 0.1
        
        # 4. 基于等待时间的权重（等待时间越长，优先级越高）
        wait_time = current_time - job.completed_time
        wait_score = wait_time * 0.5
        
        total_score = urgency_score + due_date_score + amount_score + wait_score
        return total_score

    def greedy_batch_selection(self, completed_jobs, current_time, batch_capacity=5):
        """贪心批次选择方法
        
        按配送商分组，然后按优先级排序，贪心地填充批次
        """
        if not completed_jobs:
            return {}
            
        batches = {}
        batch_id = 1
        
        # 按配送商分组
        jobs_by_distributor = self.group_by_distributor(completed_jobs)
        
        for distributor_id, jobs in jobs_by_distributor.items():
            # 计算优先级评分并排序（分数高的在前）
            sorted_jobs = sorted(jobs, 
                            key=lambda j: self.calculate_priority_score(j, current_time),
                            reverse=True)
            
            current_batch = []
            current_batch_size = 0
            
            for job in sorted_jobs:
                if current_batch_size < batch_capacity:
                    current_batch.append(job.job_id)
                    current_batch_size += 1
                else:
                    # 当前批次已满，创建新批次
                    if current_batch:
                        batches[batch_id] = current_batch
                        batch_id += 1
                    current_batch = [job.job_id]
                    current_batch_size = 1
            
            # 处理最后一个批次
            if current_batch:
                batches[batch_id] = current_batch
                batch_id += 1
        
        return batches
