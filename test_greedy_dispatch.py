#!/usr/bin/env python3
"""
测试贪心派送算法的简单脚本
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'claude_code'))

from dispatch_heuristic import DispatchHeuristic
from data_structures import Job, Operation

def create_test_jobs():
    """创建测试作业数据"""
    jobs = []
    
    # 创建一些测试作业
    job1 = Job(
        job_id=1,
        operations=[Operation(1, 1, [1, 2], {1: 5, 2: 6})],
        amount=10,
        distributor_id=1,
        due_date=20,
        completed_time=15,
        is_urgent=False
    )
    
    job2 = Job(
        job_id=2,
        operations=[Operation(2, 2, [1, 2], {1: 4, 2: 5})],
        amount=15,
        distributor_id=1,
        due_date=25,
        completed_time=18,
        is_urgent=True  # 紧急作业
    )
    
    job3 = Job(
        job_id=3,
        operations=[Operation(3, 3, [1, 2], {1: 6, 2: 7})],
        amount=8,
        distributor_id=2,
        due_date=15,  # 即将超期
        completed_time=10,
        is_urgent=False
    )
    
    job4 = Job(
        job_id=4,
        operations=[Operation(4, 4, [1, 2], {1: 3, 2: 4})],
        amount=12,
        distributor_id=2,
        due_date=30,
        completed_time=20,
        is_urgent=False
    )
    
    # 设置作业状态为已完成
    for job in [job1, job2, job3, job4]:
        job.status = 'completed'
    
    return [job1, job2, job3, job4]

def test_greedy_method():
    """测试贪心方法"""
    print("=== 测试贪心派送算法 ===\n")
    
    # 创建调度器
    dispatcher = DispatchHeuristic()
    
    # 创建测试作业
    test_jobs = create_test_jobs()
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
    
    # 测试完整的贪心派送决策
    print("\n3. 完整的贪心派送决策:")
    state = {
        'completed_jobs': test_jobs,
        't': current_time
    }
    greedy_action = dispatcher.select_action(state, method='greedy')
    print(f"贪心派送动作: {greedy_action}")
    
    print("\n" + "="*50)
    
    # 测试成本派送决策（对比）
    print("\n4. 成本派送决策 (对比):")
    cost_action = dispatcher.select_action(state, method='cost_based')
    print(f"成本派送动作: {cost_action}")
    
    print("\n" + "="*50)
    print("测试完成!")

if __name__ == "__main__":
    test_greedy_method()
