from dataclasses import dataclass
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import pandas as pd
import logging
from entity.config import Config
from .utils import EnvUtils
from entity.job_shop_entities import Job, MachineStatus, Operation, Machine, DeliveryRequirement, DistributorAssignment

@dataclass
class EnvironmentState:
    """环境状态"""
    current_time: float
    jobs: List[Job]
    machines: List[Machine]
    batches: List[Dict]
    last_schedule_time: float = 0
    num_pending_jobs: int = 0

    @classmethod
    def from_dict(cls, state_dict: Dict):
        """从字典创建状态对象"""
        return cls(
            current_time=state_dict['current_time'],
            jobs=state_dict['jobs'],
            machines=state_dict['machines'],
            batches=state_dict['batches'],
            last_schedule_time=state_dict.get('last_schedule_time', 0),
            num_pending_jobs=state_dict.get('num_pending_jobs', 0)
        )
    
class WarehouseEnvironment:
    """仓储-配送环境"""
    
    def __init__(self, config: Config, case: Dict):
        self.config = config
        self.case = case
        self.job_id_counter = 0
        self.arrival_probability = config.problem.arrival_probability
        
        # 初始化数据
        self.processing_times = case['processing_times']
        self.delivery_requirements = case['delivery_requirements']
        self.available_machines = case['available_machines']
        self.loading_times = config.problem.batch_loading_base_time
        self.per_item_times = config.problem.item_loading_time
        
        # 初始化日志
        self._setup_logger()
        
        # 重置环境
        self.reset()
    
    def _setup_logger(self):
        """设置日志"""
        self.logger = logging.getLogger('warehouse_env')
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('warehouse_env.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(handler)


    def reset(self) -> EnvironmentState:
        """重置环境状态
        
        Returns:
            EnvironmentState: 重置后的初始环境状态
        """
        # 重置作业相关状态
        self.job_id_counter = 0
        self.jobs: List[Job] = []
        self.available_jobs: List[Job] = []
        self.completed_jobs: List[Job] = []
        
        # 重置配送相关状态
        self.completed_batches: List[Dict] = []
        self.batches: List[Dict] = []
        
        # 初始化机器
        self.machines = [
            Machine(
                machine_id=m,
                capabilities=[],
                status=MachineStatus.IDLE,  # 显式设置初始状态
                total_processing_time=0.0,
                total_idle_time=0.0,
                total_setup_time=0.0
            ) for m in range(self.config.problem.num_machines)
        ]
        
        # 重置统计指标
        self.current_time = 0
        self.total_tardiness = 0
        self.machine_utilization = []
        self.batch_utilization = []
        self.last_schedule_time = 0  # 添加最后调度时间
        
        # 创建并返回初始状态
        initial_state = EnvironmentState(
            current_time=self.current_time,
            jobs=self.jobs.copy(),  # 创建副本避免引用问题
            machines=self.machines.copy(),
            batches=self.batches.copy(),
            last_schedule_time=self.last_schedule_time,
            num_pending_jobs=0  # 初始时没有待处理作业
        )
        
        return initial_state
    
    # 3. 修复 _get_state 方法
    def _get_state(self) -> Dict:
        """获取环境状态"""
        return {
            'current_time': self.current_time,
            'jobs': self.jobs,
            'machines': self.machines,
            'batches': self.batches,
            'last_schedule_time': getattr(self, 'last_schedule_time', 0),
            'num_pending_jobs': len(self.available_jobs)
        }


    def _generate_new_job(self) -> Job:
        """生成新的动态作业"""
        self.job_id_counter += 1
        
        # 生成工序
        operations = []
        num_operations = np.random.randint(
            self.config.problem.min_operations,
            self.config.problem.max_operations + 1
        )
        
        for op_id in range(num_operations):
            # 为工序分配机器
            num_machines = np.random.randint(
                1, self.config.problem.num_machines + 1
            )
            available_machines = np.random.choice(
                self.config.problem.num_machines,
                size=num_machines,
                replace=False
            ).tolist()
            
            # 创建工序对象
            operation = Operation(
                operation_id=op_id,
                available_machines=available_machines,
                processing_times={
                    m: np.random.randint(
                        self.config.problem.min_processing_time,
                        self.config.problem.max_processing_time + 1
                    )
                    for m in available_machines
                }
            )
            operations.append(operation)
        
        # 创建作业对象
        distributor_id = np.random.randint(0, self.config.problem.num_distributors)
        due_date = (self.current_time + 
                   num_operations * self.config.problem.max_processing_time + 
                   self.config.problem.due_window)
        
        return Job(
            job_id=self.job_id_counter,
            operations=operations,
            distributor_id=distributor_id,
            arrival_time=self.current_time
        )

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
        next_state = EnvironmentState.from_dict(self._get_state())  # 修改这里
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
                
                machine.current_job = j_id
                machine.remaining_time = processing_time
                job['status'] = 'processing'
                rewards += 1
            else:
                rewards -= 0.5
        return rewards

    def _update_machine_states(self):
        """更新机器状态"""
        for machine in self.machines:
            if machine.current_job is not None:
                machine.remaining_time -= 1
                machine.total_busy_time += 1
                if machine.remaining_time <= 0:
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
        """计算总延迟时间
        
        按配送商分组计算作业延迟，考虑不同配送商的交付要求
        """
        # 按配送商ID分组的作业字典
        jobs_by_distributor = defaultdict(list)
        for job in self.jobs:
            if job.complete_time is not None:  # 只考虑已完成的作业
                jobs_by_distributor[job.distributor_id].append(job)
        
        total_tardiness = 0
        # 遍历每个配送商的作业组
        for dist_id, jobs in jobs_by_distributor.items():
            # 获取该配送商的交付要求
            delivery_reqs = [
                req for req in self.delivery_requirements.itertuples()
                if req.distributor_id == dist_id
            ]
            
            # 按交付要求计算延迟
            for req in delivery_reqs:
                # 计算应交付的作业数量
                required_jobs = int(len(jobs) * req.ratio)
                if required_jobs == 0:
                    continue
                    
                # 按完成时间排序，取前 required_jobs 个作业
                sorted_jobs = sorted(jobs, key=lambda j: j.complete_time)
                relevant_jobs = sorted_jobs[:required_jobs]
                
                # 计算这批作业的延迟
                batch_tardiness = sum(
                    max(0, job.complete_time - req.due_time) * req.weight
                    for job in relevant_jobs
                )
                total_tardiness += batch_tardiness
        
        return total_tardiness


    def _calculate_machine_utilization(self):
        """计算机器利用率"""
        if self.current_time == 0:
            return 0
        return sum(m.total_busy_time for m in self.machines) / (self.current_time * len(self.machines))

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
                    f"Time {self.current_time}: Dynamic job {new_job.job_id} arrived "
                    f"with {len(new_job.operations)} operations"
                )
