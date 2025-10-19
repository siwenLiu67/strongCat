"""
环境模型模块
包含仓储-配送环境的实现
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from itertools import chain

from ..config import Config
from ..data_loader import DataLoader
from ..data_structures import Job, Operation, Machine, Distributor, DeliveryRequirement


class WarehouseEnvironment:
    """仓储-配送环境"""
    
    def __init__(self, config: Config, data_loader: DataLoader):
        """初始化仓储-配送环境
    
        Args:
            config: 配置对象
            data_loader: 数据加载器实例
        """
        # 保存配置和数据加载器
        self.config = config
        self.data_loader = data_loader
        
        # 初始化时间和状态
        self.t = 0  
        self.done = False
        
        # 初始化作业相关列表
        self.initial_jobs = []
        self.available_jobs = []
        # 加工完成的作业列表
        self.completed_jobs = []
        # 加工完成且配送完成的作业列表
        self.dispatched_jobs = []
        self.next_job_id = 0
        
        # 初始化机器和配送商
        self.machines = []
        self.distributors = []
        self.batches = []
        
        # 初始化事件和统计指标
        self.arrival_events = []
        self.total_weighted_tardiness = 0
        self.tardy_penalty = 0
        self.machine_utilization = []
        self.completion_times = {}
        self.last_schedule_time = 0
        self.last_batch_time = 0

        # 本时间步的状态
        self.operation_completed_this_step = False
        self.job_completed_this_step = False
        self.job_dispatched_this_step = False
        self.job_dispatching_this_step = False

        # 动态作业管理
        self.remaining_dynamic_jobs = config.num_dynamic_jobs
        self.dynamic_jobs_arrived = 0
        self.arrival_events = []
    
        # 执行重置
        self.reset()

    def reset(self) -> Dict:
        """重置环境"""
        self.t = 0
        self.tardy_penalty = 0
        self.done = False

        # 加载初始数据
        self.initial_jobs = self.data_loader.load_jobs()
        self.available_jobs = self.initial_jobs.copy()
        self.machines = self.data_loader.load_machines()
        self.distributors = self.data_loader.load_distributors()
        
        # 重设作业状态
        for job in self.available_jobs:
            job.status = 'waiting'
            job.current_operation = 0
            job.job_type = 'initial'
        
        # 清空已完成和已配送的作业
        self.completed_jobs = []
        self.dispatched_jobs = []

        # 重置动态到达计数器
        self.remaining_dynamic_jobs = self.config.num_dynamic_jobs
        self.dynamic_jobs_arrived = 0
        self.arrival_events = []

        # 初始化统计指标
        self.total_weighted_tardiness = 0
        self.machine_utilization = []
        self.completion_times = {}
        self.last_schedule_time = 0
        self.last_batch_time = 0

        # 设置下一个作业ID
        self.next_job_id = len(self.available_jobs)

        # 只在训练模式下显示重置信息
        if hasattr(self.config, 'train_mode') and self.config.train_mode:
            print(f"🏭 环境重置完成: 初始工件{len(self.available_jobs)}个, 待到达动态工件{self.remaining_dynamic_jobs}个")
        
        return self._get_state()

    def _log_state_transition(self, action: Dict, reward: float):
        """记录状态转换（增强版）"""
        print(f"\n{'='*20}")
        print(f"时间步 {self.t}")
        
        # 1. 作业状态（区分初始和动态作业）
        processing_jobs = [j for j in self.available_jobs if j.status == 'processing']
        initial_jobs = [j for j in self.available_jobs if getattr(j, 'job_type', 'initial') == 'initial']
        dynamic_jobs = [j for j in self.available_jobs if getattr(j, 'job_type', 'initial') == 'dynamic']
        
        print("作业状态:")
        print(f"- 总可用作业数: {len(self.available_jobs)} (初始:{len(initial_jobs)}, 动态:{len(dynamic_jobs)})")
        print(f"- 已加工作业数: {len(self.completed_jobs)}")
        print(f"- 已配送作业数: {len(self.dispatched_jobs)}")
        print(f"- 加工中作业数: {len(processing_jobs)}")
        print(f"- 配送中作业数：{len([j for j in self.available_jobs if j.status == 'dispatching'])}")
        print(f"- 等待中作业数：{len([j for j in self.available_jobs if j.status == 'waiting'])}")
    
        print(f"- 动态作业进度: {self.dynamic_jobs_arrived}/{self.config.num_dynamic_jobs}")
        
        # 2. 机器状态
        busy_machines = [m for m in self.machines if m.status == 'busy']
        print("\n机器状态:")
        print(f"- 总机器数: {len(self.machines)}")
        print(f"- 忙碌机器数: {len(busy_machines)}")
        print(f"- 机器利用率: {self.calculate_machine_utilization():.2%}")
        
        # 动作摘要
        if 'wait' in action:
            print("动作: 等待")
        elif 'schedule' in action:
            scheduled_count = len(action['schedule'])
            print(f"动作: 调度 {scheduled_count} 个作业")
        elif 'dispatch' in action:
            dispatched_count = sum(len(jobs) for jobs in action['dispatch'].values())
            print(f"动作: 配送 {dispatched_count} 个作业")
        
        # 关键事件
        if self.operation_completed_this_step:
            print("事件: 工序完成")
        if self.job_completed_this_step:
            print("事件: 作业完成")
        if self.job_dispatched_this_step:
            print("事件: 作业配送完成")
            
        print(f"奖励: {reward:.1f} | 利用率: {self.calculate_machine_utilization():.0%}")

    def _update_distributor_job_mapping(self):
        """更新配送商与作业的映射关系"""
        for distributor in self.distributors:
            # 清空当前映射
            distributor.assigned_jobs = []
            distributor.total_amount = 0
            for job in self.available_jobs:
                if job.distributor_id == distributor.distributor_id:
                    distributor.assigned_jobs.append(job.job_id)
                    distributor.total_amount += job.amount

    def _handle_batch_dynamic_arrivals(self):
        """处理批量动态作业随机到达"""
        if self.remaining_dynamic_jobs <= 0:
            return  # 所有动态作业已到达
        
        if self.t == 0:
            return  # 第一个时间步不生成动态作业
        
        # 随机决定是否有作业到达
        arrival_probability = getattr(self.config, 'batch_arrival_probability', 0.75)
        
        if np.random.random() < arrival_probability:
            # 随机决定本次到达的作业数量
            min_batch_size = getattr(self.config, 'min_batch_size', 1)
            max_batch_size = min(
                self.remaining_dynamic_jobs,
                getattr(self.config, 'max_batch_size', 5)
            )
            
            if max_batch_size >= min_batch_size:
                # 随机生成批量大小
                batch_size = np.random.randint(min_batch_size, max_batch_size + 1)
                
                arrived_jobs = []
                for i in range(batch_size):
                    if self.remaining_dynamic_jobs > 0:
                        new_job_id = self.next_job_id
                        
                        # 生成新作业
                        new_job = self.data_loader.generate_dynamic_job(new_job_id)
                        new_job.arrival_time = self.t
                        new_job.job_type = 'dynamic'
                        new_job.status = 'waiting'
                        new_job.current_operation = 0

                        self._update_distributor_job_mapping()
                        
                        # 添加到可用作业列表
                        self.available_jobs.append(new_job)
                        arrived_jobs.append(new_job)
                        
                        self.next_job_id += 1
                        # 更新计数器
                        self.remaining_dynamic_jobs -= 1
                        self.dynamic_jobs_arrived += 1
                        
                        # 记录到达事件
                        self.arrival_events.append({
                            'time': self.t,
                            'job_id': new_job.job_id,
                            'job_type': 'dynamic_batch',
                            'batch_arrival': True
                        })
                
                if arrived_jobs:
                    # 只在训练模式下显示动态作业到达信息
                    if hasattr(self.config, 'train_mode') and self.config.train_mode:
                        job_ids = [job.job_id for job in arrived_jobs]
                        print(f"⬇️ 时间步 {self.t}: 批量到达 {len(arrived_jobs)} 个动态作业 {job_ids}")
                        print(f"   📊 动态作业进度: {self.dynamic_jobs_arrived}/{self.config.num_dynamic_jobs} "
                            f"(剩余: {self.remaining_dynamic_jobs})")

    def get_dynamic_arrival_statistics(self):
        """获取动态作业到达统计信息"""
        # 按时间步统计到达情况
        arrival_by_time = {}
        for event in self.arrival_events:
            time_step = event['time']
            if time_step not in arrival_by_time:
                arrival_by_time[time_step] = 0
            arrival_by_time[time_step] += 1
        
        # 计算到达率和批次统计
        total_batches = len(set(event['time'] for event in self.arrival_events if event.get('batch_arrival', False)))
        avg_batch_size = self.dynamic_jobs_arrived / max(total_batches, 1) if total_batches > 0 else 0
        
        return {
            'total_dynamic_jobs_planned': self.config.num_dynamic_jobs,
            'dynamic_jobs_arrived': self.dynamic_jobs_arrived,
            'remaining_dynamic_jobs': self.remaining_dynamic_jobs,
            'arrival_completion_rate': self.dynamic_jobs_arrived / max(self.config.num_dynamic_jobs, 1),
            'total_arrival_events': len(self.arrival_events),
            'total_batches': total_batches,
            'avg_batch_size': round(avg_batch_size, 2),
            'arrival_by_time_step': arrival_by_time,
            'current_total_jobs': len(self.available_jobs) + len(self.completed_jobs) + len(self.dispatched_jobs),
            'initial_jobs_count': len(self.initial_jobs)
        }
    
    def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
        """执行环境步进"""
        # 重置本时间步的状态标志
        self.operation_completed_this_step = False
        self.job_completed_this_step = False
        self.job_dispatched_this_step = False
        self.job_dispatching_this_step = False
        
        # 1. 时间步开始时的状态更新
        self._update_machine_states()     # 首先更新机器状态
        self._update_job_states()         # 更新作业状态
        self._update_dispatching_jobs()  # 更新配送状态
        
        # 2. 处理动态到达（使用新的随机批量到达机制）
        self._handle_batch_dynamic_arrivals()
        
        # 3. 执行决策动作
        if 'wait' in action:
            self._process_waiting(action['wait'])
        elif 'schedule' in action:
            self._process_scheduling(action['schedule'])
            self.last_schedule_time = self.t
        elif 'dispatch' in action:
            self._process_dispatching(action['dispatch'])
            self.last_batch_time = self.t
        
        # 4. 计算奖励
        reward = self._calculate_reward(action)
        
        # 5. 检查终止条件
        self.done = self._check_termination()
        
        # 6. 记录和日志
        self._log_state_transition(action, reward)
        
        # 7. 时间步进
        self.t += 1
        
        return self._get_state(), reward, self.done, {}
    
    def _check_termination(self) -> bool:
        """检查是否达到终止条件"""
        
        # 计算预期的总作业数量
        total_expected_jobs = len(self.initial_jobs) + self.config.num_dynamic_jobs
        
        # 终止条件：
        # 1. 达到最大时间步数，或
        # 2. 所有作业（初始+动态）都已配送完成且没有剩余动态作业
        boolean_condition = self.t >= self.config.max_time_steps or (len(self.dispatched_jobs) >= total_expected_jobs and self.remaining_dynamic_jobs <= 0)
        
        # 只在达到终止条件且处于训练模式时显示信息
        if boolean_condition and hasattr(self.config, 'train_mode') and self.config.train_mode:
            print(f"🏁 达到终止条件：时间步 {self.t}/{self.config.max_time_steps} | "
                  f"已配送 {len(self.dispatched_jobs)}/{total_expected_jobs} 作业 | "
                  f"剩余动态作业 {self.remaining_dynamic_jobs}")

        return boolean_condition

    def _calculate_reward(self, action) -> float:
        reward = 0.0
        debug_info = {}
    
        # 基础动作奖励
        if 'wait' in action:
            reward -= 1.0  # 增加等待动作的惩罚
            debug_info['wait_penalty'] = -1.0
        elif 'schedule' in action:
            reward += 0.2  # 增加调度动作的奖励
            debug_info['schedule_base'] = 0.2
        elif 'dispatch' in action:
            reward += 0.4  # 增加配送动作的奖励
            debug_info['dispatch_base'] = 0.4
    
        # 调度质量奖励
        if 'schedule' in action:
            utilization = self.calculate_machine_utilization()
            reward += 0.1 * utilization
            debug_info['utilization'] = 0.1 * utilization
    
            job_progress = self.calculate_operation_progress_ratio()
            reward += 0.3 * job_progress  # 增加作业推进奖励
            debug_info['job_progress'] = 0.3 * job_progress
    
            load_balance = self.calculate_machine_load_variance()
            reward -= 0.5 * load_balance
            debug_info['load_balance_penalty'] = -0.5 * load_balance
    
        # 配送质量奖励
        if 'dispatch' in action:
            reward += 0.5 # 增加配送动作的奖励
            debug_info['dispatch_reward'] = 0.5

        # 计算作业延误（只计算一次，避免重复计算）
        current_tardiness = 0
        for job in self.completed_jobs:
            tardiness = max(0, job.dispatched_time - job.due_date)
            current_tardiness += tardiness
            debug_info[f'job_{job.job_id}_tardiness'] = -tardiness
        
        # 更新总延误（避免重复累加）
        self.total_weighted_tardiness = current_tardiness
        # 归一化total_weighted_tardiness
        reward += 1/(1 + self.total_weighted_tardiness)
        
        # 计算分段配送时间要求延迟成本（只计算配送商要求的惩罚，不重复计算作业延误）
        current_tardy_penalty = 0
        for distributor in self.distributors:
            min_due_time = min(distributor.delivery_requirements.due_times) if distributor.delivery_requirements else float('inf')
            if min_due_time < self.t:
                # 计算每个配送商的延迟成本
                requirement = distributor.delivery_requirements
                for due_time, ratio, weight in zip(requirement.due_times, requirement.ratios, requirement.weights):
                    # 在这个due_time之前完成的作业
                    completed_jobs = [j for j in self.completed_jobs if j.dispatched_time <= due_time]
                    completed_amount = sum(j.amount for j in completed_jobs)
                    required_amount = ratio * distributor.total_amount
                    if completed_amount < required_amount:
                        penalty = (required_amount - completed_amount) * weight
                        penalty_reward = (required_amount - completed_amount)/required_amount * weight
                        reward -= penalty_reward
                        debug_info[f'distributor_{distributor.distributor_id}_due_time_{due_time}'] = -penalty
                        current_tardy_penalty += penalty
        
        # 更新配送时间要求惩罚（避免重复累加）
        self.tardy_penalty = current_tardy_penalty

            
            
        # 即时操作奖励
        if self.operation_completed_this_step:
            reward += 0.5  # 增加工序完成奖励
            debug_info['operation_complete'] = 0.5
    
        if self.job_completed_this_step:
            reward += 0.8  # 增加作业完成奖励
            debug_info['job_complete'] = 0.8

        # 如果有作业被配送，增加奖励
        if self.job_dispatched_this_step:
            reward += 1.0  # 增加作业配送完成奖励
            debug_info['job_dispatched'] = 1.0

        # 只在训练模式下显示调试信息
        if hasattr(self.config, 'train_mode') and self.config.train_mode and hasattr(self.config, 'debug_reward') and getattr(self.config, 'debug_reward', False):
            print(f"奖励分解: {debug_info}")
            print(f"总奖励: {reward:.2f}")

        return reward

    def _update_machine_states(self):
        """更新机器状态"""
        for machine in self.machines:
            if machine.status == 'busy':
                # 减少剩余处理时间
                machine.remaining_time -= 1
                
                # 检查是否完成当前工序
                if machine.remaining_time <= 0:
                    # 完成工序
                    current_job = machine.current_job
                    if current_job:
                        # 更新作业状态
                        current_job.current_operation += 1
                        
                        # 检查作业是否完成
                        if current_job.current_operation >= len(current_job.operations):
                            current_job.status = 'completed'
                            current_job.completion_time = self.t
                            self.completed_jobs.append(current_job)
                            self.available_jobs.remove(current_job)
                            self.job_completed_this_step = True
                        else:
                            current_job.status = 'waiting'
                        
                        # 重置机器状态
                        machine.status = 'idle'
                        machine.current_job = None
                        machine.remaining_time = 0
                        
                        self.operation_completed_this_step = True

    def _update_job_states(self):
        """更新作业状态"""
        # 更新配送完成状态
        for job in self.completed_jobs:
            if job.status == 'completed' and job not in self.dispatched_jobs:
                # 检查配送是否完成
                if hasattr(job, 'dispatched_time') and job.dispatched_time <= self.t:
                    self.dispatched_jobs.append(job)
                    self.job_dispatched_this_step = True

    def _update_dispatching_jobs(self):
        """更新配送状态"""
        for job in self.available_jobs:
            if job.status == 'dispatching':
                # 减少配送剩余时间
                if hasattr(job, 'dispatching_remaining_time'):
                    job.dispatching_remaining_time -= 1
                    
                    # 检查配送是否完成
                    if job.dispatching_remaining_time <= 0:
                        job.status = 'dispatched'
                        job.dispatched_time = self.t
                        self.dispatched_jobs.append(job)
                        self.available_jobs.remove(job)
                        self.job_dispatched_this_step = True

    def _process_waiting(self, wait_action: Dict):
        """处理等待决策"""
        # 等待动作不需要特殊处理，只是让时间前进
        pass

    def _process_scheduling(self, schedule_action: Dict):
        """处理调度决策"""
        for job_id, machine_id in schedule_action.items():
            # 查找作业和机器
            job = next((j for j in self.available_jobs if j.job_id == job_id), None)
            machine = next((m for m in self.machines if m.machine_id == machine_id), None)
            
            if job and machine and machine.status == 'idle' and job.status == 'waiting':
                # 获取当前工序的处理时间
                current_op_index = job.current_operation
                if current_op_index < len(job.operations):
                    processing_time = job.operations[current_op_index].processing_time
                    
                    # 分配作业到机器
                    machine.status = 'busy'
                    machine.current_job = job
                    machine.remaining_time = processing_time
                    
                    # 更新作业状态
                    job.status = 'processing'
                    job.assigned_machine = machine_id

    def _process_dispatching(self, dispatch_action: Dict):
        """处理配送决策"""
        for distributor_id, job_ids in dispatch_action.items():
            distributor = next((d for d in self.distributors if d.distributor_id == distributor_id), None)
            
            if distributor:
                for job_id in job_ids:
                    job = next((j for j in self.completed_jobs if j.job_id == job_id), None)
                    
                    if job and job.status == 'completed':
                        # 开始配送
                        job.status = 'dispatching'
                        job.distributor_id = distributor_id
                        
                        # 设置配送时间（可以根据配送商属性调整）
                        dispatching_time = getattr(distributor, 'dispatching_time', 1)
                        job.dispatching_remaining_time = dispatching_time
                        
                        self.job_dispatching_this_step = True

    def _get_state(self) -> Dict:
        """获取当前状态"""
        state = {}
        
        # 机器状态
        state['machine_states'] = []
        for machine in self.machines:
            machine_state = {
                'machine_id': machine.machine_id,
                'status': machine.status,
                'remaining_time': machine.remaining_time,
                'current_job_id': machine.current_job.job_id if machine.current_job else None
            }
            state['machine_states'].append(machine_state)
        
        # 作业状态
        state['job_states'] = []
        for job in self.available_jobs:
            job_state = {
                'job_id': job.job_id,
                'status': job.status,
                'current_operation': job.current_operation,
                'total_operations': len(job.operations),
                'due_date': job.due_date,
                'weight': job.weight,
                'amount': job.amount,
                'job_type': getattr(job, 'job_type', 'initial')
            }
            state['job_states'].append(job_state)
        
        # 配送商状态
        state['distributor_states'] = []
        for distributor in self.distributors:
            distributor_state = {
                'distributor_id': distributor.distributor_id,
                'assigned_jobs': distributor.assigned_jobs,
                'total_amount': distributor.total_amount
            }
            state['distributor_states'].append(distributor_state)
        
        # 全局状态
        state['time'] = self.t
        state['completed_jobs_count'] = len(self.completed_jobs)
        state['dispatched_jobs_count'] = len(self.dispatched_jobs)
        state['available_jobs_count'] = len(self.available_jobs)
        state['remaining_dynamic_jobs'] = self.remaining_dynamic_jobs
        state['dynamic_jobs_arrived'] = self.dynamic_jobs_arrived
        
        # 性能指标
        state['machine_utilization'] = self.calculate_machine_utilization()
        state['operation_progress_ratio'] = self.calculate_operation_progress_ratio()
        state['machine_load_variance'] = self.calculate_machine_load_variance()
        state['total_weighted_tardiness'] = self.total_weighted_tardiness
        state['tardy_penalty'] = self.tardy_penalty
        
        return state

    def calculate_machine_utilization(self) -> float:
        """计算机器利用率"""
        if not self.machines:
            return 0.0
        
        busy_machines = sum(1 for machine in self.machines if machine.status == 'busy')
        return busy_machines / len(self.machines)

    def calculate_operation_progress_ratio(self) -> float:
        """计算作业生产推进进度比例"""
        if not self.available_jobs:
            return 0.0
        
        total_progress = 0
        for job in self.available_jobs:
            if hasattr(job, 'current_operation') and hasattr(job, 'operations'):
                progress = job.current_operation / max(len(job.operations), 1)
                total_progress += progress
        
        return total_progress / len(self.available_jobs)

    def calculate_machine_load_variance(self) -> float:
        """归一化所有机器的剩余时间方差"""
        if not self.machines:
            return 0.0
        
        # 收集所有机器的剩余时间
        remaining_times = []
        for machine in self.machines:
            if machine.status == 'busy':
                remaining_times.append(machine.remaining_time)
            else:
                remaining_times.append(0)
        
        if not remaining_times:
            return 0.0
        
        # 计算方差
        mean_remaining = sum(remaining_times) / len(remaining_times)
        variance = sum((x - mean_remaining) ** 2 for x in remaining_times) / len(remaining_times)
        
        # 归一化到[0,1]范围
        max_possible_variance = max(remaining_times) ** 2 if remaining_times else 1
        normalized_variance = variance / max(max_possible_variance, 1)
        
        return normalized_variance

    def get_statistics(self) -> Dict:
        """获取环境统计信息"""
        return {
            'total_time_steps': self.t,
            'initial_jobs_count': len(self.initial_jobs),
            'dynamic_jobs_arrived': self.dynamic_jobs_arrived,
            'completed_jobs_count': len(self.completed_jobs),
            'dispatched_jobs_count': len(self.dispatched_jobs),
            'available_jobs_count': len(self.available_jobs),
            'total_weighted_tardiness': self.total_weighted_tardiness,
            'tardy_penalty': self.tardy_penalty,
            'machine_utilization': self.calculate_machine_utilization(),
            'operation_progress_ratio': self.calculate_operation_progress_ratio(),
            'machine_load_variance': self.calculate_machine_load_variance(),
            'is_done': self.done
        }
