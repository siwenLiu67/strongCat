import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import pandas as pd
import logging
from data.caseBuilder.config import Config
from .utils import EnvUtils

class WarehouseEnvironment:
    """仓储-配送环境
    
    实现了分层强化学习环境，包含作业调度和配送批次规划两个子问题
    
    特点:
    - 动态作业到达和加工
    - 灵活的机器分配
    - 基于时间窗口的批次规划
    - 考虑容量和装载率约束
    """
    
    def __init__(self, config, case):
        self.config = config
        self.case = case
        self.job_id_counter = 0
        self.arrival_probability = config.problem.arrival_probability
        self.processing_times = case.get_processing_times_df()
        self.delivery_requirements = case.get_delivery_requirements_df()
        self.available_machines = case.get_available_machines_df()
        self.loading_times = config.problem.batch_loading_base_time
        self.per_item_times = config.problem.item_loading_time
        
        self.logger = logging.getLogger('warehouse_env')
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('warehouse_env.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(handler)
        self.reset()

    def reset(self):
        """重置环境状态"""
        self.job_id_counter = 0
        self.jobs = []
        self.available_jobs = []
        self.completed_jobs = []
        self.completed_batches = []
        self.machines = EnvUtils.initialize_machines(self.config, self.available_machines)
        self.batches = EnvUtils.initialize_batches(self.delivery_requirements)
        self.current_time = 0
        self.total_tardiness = 0
        self.machine_utilization = []
        self.batch_utilization = []
        return self._get_state()

    def _initialize_jobs(self):
        """初始化作业加工时间和工序信息"""
        jobs = []
        for job_id, job_data in self.processing_times.groupby('job_id'):
            operations = []
            for op_id in job_data['operation'].drop_duplicates():
                op_data = job_data[job_data['operation'] == op_id]
                operations.append({
                    'operation_id': op_id,
                    'machines': op_data['machine'].tolist(),
                    'processing_time': op_data['processing_time'].values[0] if not op_data.empty else 0
                })
            
            distributor_id = job_data['distributor_id'].drop_duplicates().values[0] if 'distributor_id' in job_data else None
            delivery_requirements = self.delivery_requirements[self.delivery_requirements['distributor_id'] == distributor_id]
            
            jobs.append({
                'id': job_id,
                'operations': operations,
                'current_op': 0,
                'status': 'waiting',
                'start_time': None,
                'complete_time': None,
                'delivery_requirements': delivery_requirements,
                'distributor': distributor_id
            })
        return jobs

    def _is_valid_batch_assignment(self, job, batch):
        """检查批次分配是否可行"""
        if job['distributor'] != batch['distributor']:
            return False
        if job['complete_time'] is not None and job['complete_time'] > batch['due_time']:
            return False
        return True

    def _process_batching(self, batch_assignment):
        """处理配送批次决策"""
        rewards = 0
        for j_id, b_id in batch_assignment.items():
            if b_id is None:
                continue
            job = self.jobs[j_id]
            batch = next(b for b in self.batches if b['id'] == b_id)
            if self._is_valid_batch_assignment(job, batch):
                batch['assigned_jobs'].append(j_id)
                rewards += 1
            else:
                rewards -= 0.5
        return rewards

    def _complete_operation(self, machine):
        """完成当前工序"""
        job_id = machine['current_job']
        if job_id is None:
            return
            
        job = self.jobs[job_id]
        current_op = job['operations'][job['current_op']]
        current_op['status'] = 'completed'
        current_op['complete_time'] = self.current_time
        
        job['current_op'] += 1
        if job['current_op'] >= len(job['operations']):
            job['status'] = 'completed'
            job['complete_time'] = self.current_time
            self.completed_jobs.append(job_id)
        else:
            job['status'] = 'waiting'
        
        machine['current_job'] = None
        machine['remaining_time'] = 0

    def step(self, actions):
        """执行环境步进"""
        if self.current_time > 0:
            self._process_dynamic_arrivals()

        schedule_rewards = self._process_scheduling(actions['machine_assignment'])
        batch_rewards = self._process_batching(actions['batch_assignment'])
        
        self.current_time += 1
        self._update_machine_states()
        self._update_batch_states()
        
        reward = self._calculate_reward(schedule_rewards, batch_rewards)
        done = self._check_termination()
        next_state = self._get_state()
        info = self._get_step_info()
        
        return next_state, reward, done, info

    def _process_scheduling(self, machine_assignment):
        """处理作业调度决策"""
        rewards = 0
        for m_id, j_id in enumerate(machine_assignment):
            if j_id is None:
                continue
                
            machine = self.machines[m_id]
            job = self.jobs[j_id]
            
            if EnvUtils.is_valid_assignment(job, machine):
                processing_time = self.processing_times.loc[
                    (self.processing_times['job_id'] == j_id) &
                    (self.processing_times['machine'] == m_id),
                    'processing_time'
                ].values[0]
                
                machine['current_job'] = j_id
                machine['remaining_time'] = processing_time
                job['status'] = 'processing'
                rewards += 1
            else:
                rewards -= 0.5
        return rewards

    def _update_machine_states(self):
        """更新机器状态"""
        for machine in self.machines:
            if machine['current_job'] is not None:
                machine['remaining_time'] -= 1
                machine['total_busy_time'] += 1
                if machine['remaining_time'] <= 0:
                    self._complete_operation(machine)

    def _update_batch_states(self):
        """更新批次状态"""
        current_time = self.current_time
        for batch in self.batches:
            if batch['start_time'] is None:
                advance = getattr(self.config, 'batch_start_advance', 0)
                batch['start_time'] = max(0, batch['due_time'] - advance)
            
            if batch.get('end_time', None) is None:
                batch['end_time'] = batch['due_time']

            if batch['status'] == 'waiting' and current_time >= batch['start_time']:
                batch['status'] = 'active'
            elif batch['status'] == 'active' and current_time > batch['end_time']:
                min_jobs = batch.get('min_jobs', batch.get('required_jobs', 0))
                if len(batch['assigned_jobs']) >= min_jobs:
                    batch['status'] = 'completed'
                    self.completed_batches.append(batch['id'])
                else:
                    batch['status'] = 'failed'

    def _calculate_tardiness(self):
        """计算总延迟时间"""
        total_tardiness = 0
        for job in self.jobs:
            if job['complete_time'] is not None:
                tardiness = max(0, job['complete_time'] - job['due_date'])
                total_tardiness += tardiness
        return total_tardiness

    def _calculate_machine_utilization(self):
        """计算机器利用率"""
        if self.current_time == 0:
            return 0
        return sum(m['total_busy_time'] for m in self.machines) / (self.current_time * len(self.machines))

    def _calculate_batch_utilization(self):
        """计算批次利用率"""
        return EnvUtils.calculate_batch_utilization(self.batches)

    def _check_termination(self):
        """检查终止条件"""
        return (len(self.completed_jobs) == len(self.jobs) or 
                len(self.completed_batches) == len(self.batches) or
                self.current_time >= self.config.max_time_steps)

    def _get_step_info(self):
        """获取步骤信息"""
        return {
            'completed_jobs': len(self.completed_jobs),
            'completed_batches': len(self.completed_batches),
            'current_time': self.current_time,
            'tardiness': self.total_tardiness,
            'machine_utilization': self._calculate_machine_utilization(),
            'batch_utilization': self._calculate_batch_utilization()
        }

    def _calculate_reward(self, schedule_rewards, batch_rewards):
        """计算总奖励"""
        reward = 0
        reward += self.config.reward.scheduling_weight * schedule_rewards
        reward += self.config.reward.batching_weight * batch_rewards
        reward -= self.config.reward.tardiness_weight * self._calculate_tardiness()
        reward += self.config.reward.utilization_weight * (
            self._calculate_machine_utilization() + self._calculate_batch_utilization()) / 2
        return reward

    def _get_state(self):
        """获取环境状态"""
        return {
            'current_time': self.current_time,
            'jobs': self.jobs,
            'machines': self.machines,
            'batches': self.batches,
            'last_schedule_time': getattr(self, 'last_schedule_time', 0),
            'num_pending_jobs': len(self.available_jobs)
        }

    def _generate_new_job(self) -> Dict:
        """生成新的动态作业"""
        self.job_id_counter += 1
        num_operations = np.random.randint(
            self.config.problem.min_operations,
            self.config.problem.max_operations + 1
        )
        
        operations = []
        for op_id in range(num_operations):
            num_machines = np.random.randint(1, self.config.problem.num_machines + 1)
            available_machines = np.random.choice(
                self.config.problem.num_machines,
                size=num_machines,
                replace=False
            ).tolist()
            
            operations.append({
                'operation_id': op_id,
                'machines': available_machines,
                'processing_time': np.random.randint(
                    self.config.problem.min_processing_time,
                    self.config.problem.max_processing_time + 1
                ),
                'status': 'waiting',
                'complete_time': None
            })
        
        distributor_id = np.random.randint(0, self.config.problem.num_distributors)
        due_date = self.current_time + num_operations * self.config.problem.max_processing_time + self.config.problem.due_window
        
        return {
            'id': f"dyn_{self.job_id_counter}",
            'operations': operations,
            'current_op': 0,
            'status': 'waiting',
            'start_time': None,
            'complete_time': None,
            'arrival_time': self.current_time,
            'due_date': due_date,
            'distributor': distributor_id,
            'priority': np.random.randint(1, self.config.problem.max_priority + 1)
        }

    def _process_dynamic_arrivals(self):
        """处理动态作业到达"""
        if self.current_time == 0:
            return
            
        if np.random.random() < self.arrival_probability:
            num_new_jobs = np.random.poisson(self.config.problem.arrival_batch_size)
            for _ in range(num_new_jobs):
                new_job = self._generate_new_job()
                self.jobs.append(new_job)
                self.available_jobs.append(new_job)
                self.logger.info(
                    f"Time {self.current_time}: Dynamic job {new_job['id']} arrived "
                    f"with {len(new_job['operations'])} operations"
                )
