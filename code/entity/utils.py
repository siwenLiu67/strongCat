import numpy as np
from typing import Dict, List
import pandas as pd

class EnvUtils:
    """仓储环境工具类"""
    
    @staticmethod
    def initialize_machines(config, available_machines_df):
        """初始化机器状态"""
        machines = []
        machine_capabilities = available_machines_df.groupby('machine_id')['job_id'].apply(list).to_dict()
        for m_id in range(config.problem.num_machines):
            machines.append({
                'id': m_id,
                'current_job': None,
                'remaining_time': 0,
                'capabilities': machine_capabilities.get(m_id, [])
            })
        return machines

    @staticmethod
    def initialize_batches(delivery_requirements):
        """初始化配送批次要求"""
        batches = []
        for _, req in delivery_requirements.iterrows():
            batches.append({
                'id': f"{req['distributor_id']}_{req['due_time']}",
                'distributor': req['distributor_id'],
                'due_time': req['due_time'],
                'required_ratio': req['ratio'],
                'required_jobs': int(len(req['jobs']) * req['ratio']),
                'assigned_jobs': [],
                'status': 'waiting'
            })
        return batches

    @staticmethod
    def is_valid_assignment(job, machine):
        """检查作业-机器分配是否可行"""
        if machine['current_job'] is not None:
            return False
        if job['current_op'] >= len(job['operations']):
            return False
        return machine['id'] in job['operations'][job['current_op']]['machines']

    @staticmethod
    def calculate_batch_utilization(batches):
        """计算批次利用率"""
        active_batches = [b for b in batches if b['status'] != 'completed']
        if not active_batches:
            return 0
        utilizations = np.array([len(b['assigned_jobs']) for b in active_batches]) / \
                      np.array([b['required_jobs'] for b in active_batches])
        return float(utilizations.mean())
