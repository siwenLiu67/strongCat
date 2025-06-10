import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import pandas as pd
import logging

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
        """初始化环境
        
        参数:
            config: 配置对象
            case: 预生成的算例数据
        """
        self.config = config
        self.case = case
        
        # 从算例中获取基本信息
        self.processing_times = case.get_processing_times_df()
        self.delivery_requirements = case.get_delivery_requirements_df()
        self.available_machines = case.get_available_machines_df()
        
        self.loading_times = config.batch_loading_base_time
        self.per_item_times = config.item_loading_time
        
        
        # 初始化日志
        self.logger = logging.getLogger('warehouse_env')
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('warehouse_env.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(handler)

        # 初始化环境
        self.reset()
    
    def reset(self):
        """重置环境状态"""
        # 初始化作业状态
        self.jobs = self._initialize_jobs()
        
        # 初始化机器状态
        self.machines = self._initialize_machines()
        
        # 初始化配送批次状态
        self.batches = self._initialize_batches()
        
        # 重置统计指标
        self.current_time = 0
        self.completed_jobs = []
        self.completed_batches = []
        self.total_tardiness = 0
        self.machine_utilization = []
        self.batch_utilization = []
        
        return self._get_state()
    

    def _initialize_jobs(self):
        """初始化作业加工时间和工序信息
        
        从processing_times表格(job_id, operation, machine, processing_time)中
        构建作业字典，整合每个作业的所有工序信息
        """
        jobs = []
        # 按工件ID分组
        for job_id, job_data in self.processing_times.groupby('job_id'):
            # 获取该工件的所有工序信息
            operation_ids = job_data['operation'].drop_duplicates()
            # 按工序号分组
            operations = []
            for op_id in operation_ids:
                available_machines = job_data[job_data['operation'] == op_id]['machine']
                processing_times = job_data[job_data['operation'] == op_id]['processing_time']
                
                # 构建工序字典
                operation = {
                    'operation_id': op_id,
                    'machines': available_machines.tolist(),  # 可用机器列表
                    'processing_times': processing_times.values[0] if not processing_times.empty else 0  # 加工时间
                }
                operations.append(operation)
            
            distributor_id = job_data['distributor_id'].drop_duplicates().values[0] if 'distributor_id' in job_data else None

            # 找到distributor_id对应的交付要求
            delivery_requirements = self.delivery_requirements[    
                self.delivery_requirements['distributor_id'] == distributor_id]
            # 创建作业字典
            job = {
                'id': job_id,
                'operations': operations,  # 工序信息
                'current_op': 0,  # 当前工序索引
                'status': 'waiting',  # 初始状态
                'start_time': None,  # 开始时间
                'complete_time': None,  # 完成时间
                # 从其他数据源获取due_date和distributor信息
                'delivery_requirements': delivery_requirements,
                'distributor': distributor_id
            }
            jobs.append(job)
        
        return jobs
    
    
    def _initialize_machines(self):
        """初始化机器状态"""
        machines = []
        for m in range(self.config.num_machines):
            capable_jobs = []
            # 找到机器可加工的工件id,available_machines是一个DataFrame,包含job_id和capable_machines两列
            for _, row in self.available_machines.iterrows():
                for item in row['available_machines']:
                    if item == m and row['job_id'] not in capable_jobs:
                        # 如果机器可以加工该作业，则添加到capable_jobs列表
                        capable_jobs.append(row['job_id'])
                    
            machine = {
                'id': m,
                'current_job': None,
                'remaining_time': 0,
                'queue': [],
                'total_busy_time': 0,
                'state_history': [],    
                'capabilities': capable_jobs  # 机器可以加工的作业列表
            }
            machines.append(machine)
        return machines
    
    def _initialize_batches(self):
        """初始化配送批次要求
        
        每个批次要求包含:
        - 截止时间(due_time)
        - 要求完成的作业比例(required_ratio)
        - 所属配送商(distributor)
        """
        batches = []
        for _, req in self.delivery_requirements.iterrows():
            batch = {
                'id': str(req['distributor_id']) + '_' + str(req['due_time']),  # 批次ID
                'distributor': req['distributor_id'],
                'due_time': req['due_time'],           # 截止时间
                'required_ratio': req['ratio'],        # 要求完成比例
                'required_jobs': int(len(req['jobs']) * req['ratio']),  # 需要完成的作业数
                'assigned_jobs': [],
                'status': 'waiting',
                'start_time': None,
                'end_time': None,
            }
            batches.append(batch)
        return batches
    
    def _is_valid_batch_assignment(self, job, batch):
        """检查批次分配是否可行
        
        检查条件:
        1. 作业属于正确的配送商
        2. 作业完工时间在截止时间之前
        """
        # 检查配送商匹配
        if job['distributor'] != batch['distributor']:
            return False
            
        # 检查截止时间约束
        if job['complete_time'] is not None and job['complete_time'] > batch['due_time']:
            return False
            
        return True
    
    def _update_batch_states(self):
        """更新批次状态"""
        for batch in self.batches:
            if batch['status'] == 'waiting':
                # 如果有作业分配，则激活批次
                if len(batch['assigned_jobs']) > 0:
                    batch['status'] = 'active'
                    
            elif batch['status'] == 'active':
                # 检查是否达到要求的作业数量
                if len(batch['assigned_jobs']) >= batch['required_jobs']:
                    batch['status'] = 'completed'
                    self.completed_batches.append(batch['id'])
                # 检查是否超过截止时间但未达到要求
                elif self.current_time > batch['due_time']:
                    batch['status'] = 'failed'

                    
    def _process_batching(self, batch_assignment):
        """处理配送批次决策
        
        只考虑批次的作业数量约束和时间窗口约束
        """
        rewards = 0
        
        for j_id, b_id in batch_assignment.items():
            if b_id is None:
                continue
                
            job = self.jobs[j_id]
            batch = next(b for b in self.batches if b['id'] == b_id)
            
            # 检查批次分配是否可行
            if self._is_valid_batch_assignment(job, batch):
                # 执行分配
                batch['assigned_jobs'].append(j_id)
                rewards += 1
            else:
                rewards -= 0.5
                
        return rewards
    

    def _calculate_batch_utilization(self):
        """计算批次利用率（向量化版本）"""
        active_batches = [b for b in self.batches if b['status'] != 'completed']
        if not active_batches:
            return 0
            
        utilizations = np.array([len(b['assigned_jobs']) for b in active_batches]) / \
                      np.array([b['required_jobs'] for b in active_batches])
        
        return float(utilizations.mean())
    
    
    def _is_valid_assignment(self, job, machine):
        """检查作业-机器分配是否可行"""
        # 检查机器是否空闲
        if machine['current_job'] is not None:
            return False
            
        # 检查工序顺序约束
        if job['current_op'] >= len(job['operations']):
            return False
            
        current_operation = job['operations'][job['current_op']]
        
        # 检查机器是否可以处理该工序
        if machine['id'] not in current_operation['machines']:
            return False
            
        # 检查前序工序是否完成
        if job['current_op'] > 0 and job['operations'][job['current_op'] - 1]['status'] != 'completed':
            return False
            
        return True
    
    def _complete_operation(self, machine):
        """完成当前工序"""
        job_id = machine['current_job']
        if job_id is None:
            return
            
        job = self.jobs[job_id]
        current_op = job['operations'][job['current_op']]
        
        # 更新工序状态
        current_op['status'] = 'completed'
        current_op['complete_time'] = self.current_time
        
        # 更新作业状态
        job['current_op'] += 1
        if job['current_op'] >= len(job['operations']):
            job['status'] = 'completed'
            job['complete_time'] = self.current_time
            self.completed_jobs.append(job_id)
        else:
            job['status'] = 'waiting'
        
        # 重置机器状态
        machine['current_job'] = None
        machine['remaining_time'] = 0


    def step(self, actions):
        """执行环境步进
        
        参数:
            actions: 包含两部分决策
                - machine_assignment: 作业-机器分配决策
                - batch_assignment: 作业-批次分配决策
        """
        # 1. 处理作业调度决策
        schedule_rewards = self._process_scheduling(actions['machine_assignment'])
        
        # 2. 处理配送批次决策
        batch_rewards = self._process_batching(actions['batch_assignment'])
        
        # 3. 更新环境状态
        self.current_time += 1
        self._update_machine_states()
        self._update_batch_states()
        
        # 4. 计算奖励
        reward = self._calculate_reward(schedule_rewards, batch_rewards)
        
        # 5. 检查终止条件
        done = self._check_termination()
        
        # 6. 获取下一状态和信息
        next_state = self._get_state()
        info = self._get_step_info()
        
        return next_state, reward, done, info
    
    def _process_scheduling(self, machine_assignment):
        """处理作业调度决策"""
        rewards = 0
        
        # 遍历每个机器的分配决策
        for m_id, j_id in enumerate(machine_assignment):
            if j_id is None:
                continue
                
            machine = self.machines[m_id]
            job = self.jobs[j_id]
            
            # 检查分配是否可行
            if self._is_valid_assignment(job, machine):
                # 执行分配
                processing_time = self.processing_times.loc[
                    (self.processing_times['job_id'] == j_id) &
                    (self.processing_times['machine_id'] == m_id),
                    'processing_time'
                ].values[0]
                
                machine['current_job'] = j_id
                machine['remaining_time'] = processing_time
                job['status'] = 'processing'
                
                rewards += 1  # 可行分配奖励
            else:
                rewards -= 0.5  # 不可行分配惩罚
                
        return rewards
        

    def _update_machine_states(self):
        """更新机器状态"""
        for machine in self.machines:
            if machine['current_job'] is not None:
                machine['remaining_time'] -= 1
                machine['total_busy_time'] += 1
                
                # 检查作业是否完成
                if machine['remaining_time'] <= 0:
                    self._complete_operation(machine)
    
    def _update_batch_states(self):
        """更新批次状态"""
        current_time = self.current_time

        for batch in self.batches:
            # 如果start_time为None，初始化为due_time - 配置的提前量（如有），否则为0或当前时间
            if batch['start_time'] is None:
                # 你可以根据业务需求设置提前量，比如提前x步激活批次
                advance = getattr(self.config, 'batch_start_advance', 0)
                batch['start_time'] = max(0, batch['due_time'] - advance)
            # 如果end_time为None，初始化为due_time
            if batch.get('end_time', None) is None:
                batch['end_time'] = batch['due_time']

            if batch['status'] == 'waiting':
                # 检查是否到达开始时间
                if current_time >= batch['start_time']:
                    batch['status'] = 'active'

            elif batch['status'] == 'active':
                # 检查是否超过结束时间
                if current_time > batch['end_time']:
                    # 这里假设有min_jobs字段，否则用required_jobs
                    min_jobs = batch.get('min_jobs', batch.get('required_jobs', 0))
                    if len(batch['assigned_jobs']) >= min_jobs:
                        batch['status'] = 'completed'
                        self.completed_batches.append(batch['id'])
                    else:
                        # 未满足最小装载率要求
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
        
        total_busy_time = sum(m['total_busy_time'] for m in self.machines)
        return total_busy_time / (self.current_time * len(self.machines))
    
    def _check_termination(self):
        """检查是否满足终止条件"""
        # 所有作业完成
        all_jobs_completed = len(self.completed_jobs) == len(self.jobs)
        
        # 所有批次处理完成
        all_batches_completed = len(self.completed_batches) == len(self.batches)
        
        # 超时检查
        timeout = self.current_time >= self.config.max_time_steps
        
        return all_jobs_completed or all_batches_completed or timeout
    
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
        
        # 1. 调度相关奖励
        reward += self.config.scheduling_weight * schedule_rewards
        
        # 2. 批次相关奖励
        reward += self.config.batching_weight * batch_rewards
        
        # 3. 延迟惩罚
        tardiness = self._calculate_tardiness()
        reward -= self.config.tardiness_weight * tardiness
        
        # 4. 利用率奖励
        machine_util = self._calculate_machine_utilization()
        batch_util = self._calculate_batch_utilization()
        reward += self.config.utilization_weight * (machine_util + batch_util) / 2
        
        return reward
    
    def _get_state(self):
        """获取环境状态"""
        return {
            'meta': self._get_meta_state(),
            'warehouse': {
                'nodes': self._get_warehouse_nodes(),
                'adj': self._get_warehouse_adjacency()
            },
            'distribution': {
                'nodes': self._get_distribution_nodes(),
                'adj': self._get_distribution_adjacency()
            }
        }
    

    
    def render(self):
        """渲染当前环境状态"""
        print("\n" + "="*50)
        print(f"当前时间步: {self.current_time}")
        print(f"作业完成: {len(self.completed_jobs)}/{len(self.jobs)}")
        print(f"批次完成: {len(self.completed_batches)}/{len(self.batches)}")
        
        print("\n机器状态:")
        for m in self.machines:
            status = "空闲" if m['current_job'] is None else f"加工作业{m['current_job']}"
            print(f"机器{m['id']}: {status} (剩余时间: {m['remaining_time']})")
        
        print("\n活跃批次:")
        active_batches = [b for b in self.batches if b['status'] == 'active']
        for b in active_batches:
            print(f"批次{b['id']}: {len(b['assigned_jobs'])}/{b['max_jobs']} 个作业")
        
        print("\n性能指标:")
        print(f"总延迟: {self._calculate_tardiness():.2f}")
        print(f"机器利用率: {self._calculate_machine_utilization():.2f}")
        print(f"批次利用率: {self._calculate_batch_utilization():.2f}")
        print("="*50)



    def _get_meta_state(self):
        """构建元控制层状态向量
        
        Returns:
            np.ndarray: 状态向量包含以下5个全局指标:
            1. q_t: 输入缓冲区中的平均作业数量
            2. sigma_mach_t: 各机器剩余加工时间的标准差(工作负载不平衡指标)
            3. s_due_t: 缓冲区作业的平均时间裕度
            4. rho_urg_t: 缓冲区中紧急作业的比例
            5. mu_load_t: 系统中所有作业按订单类型的估计加工负载
        """
        # 1. 计算输入缓冲区平均作业数
        buffered_jobs = [job for job in self.jobs 
                        if job['status'] == 'waiting']
        q_t = len(buffered_jobs) / len(self.machines)  # 归一化到机器数量
        
        # 2. 计算机器剩余加工时间的标准差
        remaining_times = []
        for machine in self.machines:
            if machine['current_job'] is not None:
                remaining_times.append(machine['remaining_time'])
            else:
                remaining_times.append(0)
        sigma_mach_t = np.std(remaining_times) if remaining_times else 0
        
        # 归一化标准差
        max_processing_time = self.config.max_processing_time
        sigma_mach_t = sigma_mach_t / max_processing_time if max_processing_time > 0 else 0
        
        # 3. 计算缓冲区作业的平均时间裕度
        current_time = self.current_time
        slack_times = []
        for job in buffered_jobs:
            due_date = job.get('due_date', self.config.max_time_steps)
            # 估计剩余加工时间
            remaining_ops = len(job['operations']) - job['current_op']
            est_remaining_time = remaining_ops * self.config.avg_processing_time
            slack = max(0, due_date - (current_time + est_remaining_time))
            slack_times.append(slack)
        s_due_t = np.mean(slack_times) if slack_times else 0

    
        # 归一化时间裕度
        s_due_t = s_due_t / self.config.max_time_steps
        
        # 4. 计算紧急作业比例
        urgent_threshold = self.config.urgent_threshold  # 紧急时间阈值
        urgent_jobs = sum(1 for job in buffered_jobs 
                         if job.get('due_date', self.config.max_time_steps) - 
                         current_time <= urgent_threshold)
        rho_urg_t = urgent_jobs / len(buffered_jobs) if buffered_jobs else 0
        
        # 5. 计算订单类型的加工负载
        order_type_loads = defaultdict(float)
        for job in self.jobs:
            if job['status'] != 'completed':
                order_type = job.get('order_type', 'default')
                # 估算剩余加工负载
                remaining_ops = len(job['operations']) - job['current_op']
                est_load = remaining_ops * self.config.avg_processing_time
                order_type_loads[order_type] += est_load
        # 计算平均负载
        mu_load_t = np.mean(list(order_type_loads.values())) if order_type_loads else 0
        # 归一化负载
        mu_load_t = mu_load_t / (self.config.max_time_steps * len(self.machines))
        
        return np.array([
            q_t,            # 平均缓冲作业数
            sigma_mach_t,   # 机器负载标准差
            s_due_t,        # 平均时间裕度
            rho_urg_t,      # 紧急作业比例
            mu_load_t       # 订单类型平均负载
        ], dtype=np.float32).reshape(1, 1, -1)  # 确保输出形状为 [1, 1, 5]
    
    
    def _get_warehouse_nodes(self):
        """构建仓储节点特征"""
        # 作业节点特征
        job_features = []
        for job in self.jobs:
            features = [
                job['id'] / len(self.jobs),                # 归一化ID
                job['current_op'] / len(job['operations']), # 工序进度
                float(job['status'] == 'waiting'),         # 等待状态
                float(job['status'] == 'processing'),      # 加工状态
                float(job['status'] == 'completed')        # 完成状态
            ]
            job_features.append(features)
        
        # 机器节点特征
        machine_features = []
        for machine in self.machines:
            features = [
                machine['id'] / len(self.machines),        # 归一化ID
                float(machine['current_job'] is None),     # 空闲状态
                machine['remaining_time'] / self.config.max_processing_time,  # 剩余时间
                machine['total_busy_time'] / max(1, self.current_time)       # 利用率
            ]
            machine_features.append(features)
        
        return {
            'jobs': np.array(job_features, dtype=np.float32),
            'machines': np.array(machine_features, dtype=np.float32)
        }
    

    def _get_warehouse_adjacency(self):
        """
        Constructs the warehouse adjacency matrix representing the relationship between jobs and machines.
        Returns:
            np.ndarray: A 2D numpy array of shape (number of jobs, number of machines), where each entry (i, j) is 1 if job i's current operation can be processed on machine j, and 0 otherwise. Only jobs with status 'waiting' and with remaining operations are considered.
        """
        """构建仓储邻接矩阵"""
        n_jobs = len(self.jobs)
        n_machines = len(self.machines)
        adj_matrix = np.zeros((n_jobs, n_machines), dtype=np.float32)
        
        for j_idx, job in enumerate(self.jobs):
            if job['status'] == 'waiting' and job['current_op'] < len(job['operations']):
                current_op = job['operations'][job['current_op']]
                for m_id in current_op['machines']:
                    adj_matrix[j_idx, m_id] = 1
        
        return adj_matrix
    

    def _get_distribution_nodes(self):
        """构建配送节点特征"""
        # 已完工作业特征
        job_features = []
        for job in self.jobs:
            if job['status'] == 'completed' and job['complete_time'] is not None:
                features = [
                    job['id'] / len(self.jobs),                # 归一化ID
                    (job['due_date'] - self.current_time) / self.config.max_time_steps,  # 归一化剩余时间
                    job['complete_time'] / self.config.max_time_steps,  # 归一化完工时间
                    float(job['distributor']) / self.config.num_distributors  # 归一化配送商ID
                ]
                job_features.append(features)
        
        # 批次节点特征
        batch_features = []
        for batch in self.batches:
            if batch['status'] == 'completed' and len(batch['assigned_jobs']) > 0:
                features = [
                    len(batch['assigned_jobs']) / batch['required_jobs'],  # 装载率
                    batch['start_time'] / self.config.max_time_steps,  # 归一化开始时间
                    batch['end_time'] / self.config.max_time_steps,    # 归一化结束时间
                    float(batch['status'] == 'active'),               # 活跃状态
                    float(batch['status'] == 'waiting'),              # 等待状态
                    batch['priority'] / len(self.batches)             # 归一化优先级
                ]
                batch_features.append(features)
            
        return {
            'jobs': np.array(job_features, dtype=np.float32) if job_features else np.zeros((0, 4), dtype=np.float32),
            'batches': np.array(batch_features, dtype=np.float32)
        }
    

    def _get_distribution_adjacency(self):
        """构建配送邻接矩阵：已完工作业-批次关系"""
        completed_jobs = [j for j in self.jobs if j['status'] == 'completed']
        n_jobs = len(completed_jobs)
        n_batches = len(self.batches)
        
        adj_matrix = np.zeros((n_jobs, n_batches), dtype=np.float32)
        
        for j_idx, job in enumerate(completed_jobs):
            for b_idx, batch in enumerate(self.batches):
                # 判断作业是否可以分配给该批次
                if (batch['status'] in ['waiting', 'active'] and 
                    job['distributor'] == batch['distributor'] and
                    len(batch['assigned_jobs']) < batch['max_jobs']):
                    adj_matrix[j_idx, b_idx] = 1
        
        return adj_matrix