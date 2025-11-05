#!/usr/bin/env python3
"""
简单的贪心派送算法测试
"""

import sys
import os

# 直接复制需要的类定义来测试
from collections import defaultdict

class SimpleJob:
    """简化的作业类用于测试"""
    def __init__(self, job_id, distributor_id, amount, due_date, completed_time, is_urgent=False):
        self.job_id = job_id
        self.distributor_id = distributor_id
        self.amount = amount
        self.due_date = due_date
        self.completed_time = completed_time
        self.is_urgent = is_urgent
        self.status = 'completed'

class SimpleDispatchHeuristic:
    """简化的派送启发式类"""
    def __init__(self):
        pass

    def group_by_distributor(self, jobs):
        """按配送商分组作业"""
        jobs_by_distributor = defaultdict(list)
        for job in jobs:
            jobs_by_distributor[job.distributor_id].append(job)
        return jobs_by_distributor

    def calculate_priority_score(self, job, current_time):
        """计算作业的优先级评分"""
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
        """贪心批次选择方法"""
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

def test_greedy_method():
    """测试贪心方法"""
    print("=== 测试贪心派送算法 ===\n")
    
    # 创建调度器
    dispatcher = SimpleDispatchHeuristic()
    
    # 创建测试作业
    test_jobs = [
        SimpleJob(1, 1, 10, 20, 15, False),
        SimpleJob(2, 1, 15, 25, 18, True),  # 紧急作业
        SimpleJob(3, 2, 8, 15, 10, False),  # 即将超期
        SimpleJob(4, 2, 12, 30, 20, False)
    ]
    
    current_time = 20
    
    print("测试作业信息:")
    for job in test_jobs:
        priority_score = dispatcher.calculate_priority_score(job, current_time)
        print(f"作业 {job.job_id}: 配送商 {job.distributor_id}, 数量 {job.amount}, "
              f"截止时间 {job.due_date}, 紧急 {job.is_urgent}, 优先级分数 {priority_score:.2f}")
    
    print("\n" + "="*50)
    
    # 测试按配送商分组
    print("\n1. 按配送商分组:")
    grouped = dispatcher.group_by_distributor(test_jobs)
    for distributor_id, jobs in grouped.items():
        job_ids = [j.job_id for j in jobs]
        print(f"配送商 {distributor_id}: 作业 {job_ids}")
    
    print("\n" + "="*50)
    
    # 测试贪心批次选择
    print("\n2. 贪心批次选择 (批次容量=3):")
    batches = dispatcher.greedy_batch_selection(test_jobs, current_time, batch_capacity=3)
    print(f"生成的批次: {batches}")
    
    print("\n" + "="*50)
    print("测试完成!")

if __name__ == "__main__":
    test_greedy_method()
