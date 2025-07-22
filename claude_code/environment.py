from config import Config
from typing import Dict, Tuple
from data_structures import Job, Operation,Machine,Distributor,DeliveryRequirement
from case_generator import FlexibleJobShopScenario
import numpy as np
from itertools import chain

class WarehouseEnvironment:
    """仓储-配送环境"""
    
    def __init__(self, config: Config, case: FlexibleJobShopScenario):
        """初始化仓储-配送环境
    
        Args:
            config: 配置对象
            case: 算例生成器实例
        """
        # 保存配置和算例
        self.config = config
        self.case = case
        
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
        
        # 初始化机器和配送商
        self.machines = []
        self.distributors = []
        self.batches=[]
        
        # 初始化事件和统计指标
        self.arrival_events = []
        self.total_weighted_tardiness = 0
        self.machine_utilization = []
        self.completion_times = {}
        self.last_schedule_time = 0
        self.last_batch_time = 0

        # 本时间步的状态
        self.operation_completed_this_step = False
        self.job_completed_this_step = False
        self.job_dispatched_this_step = False
        self.job_dispatching_this_step = False
        
        # 执行重置
        self.reset()


    def reset(self) -> Dict:
        self.t = 0
        self.tardy_penalty = 0
        self.done = False

        # 初始化作业
        self.initial_jobs = self.case.jobs.copy()
        # 包括新到达的作业
        self.available_jobs = self.case.jobs.copy()
        # 重设作业状态
        for job in self.available_jobs:
            job.status = 'waiting'
            job.current_operation = 0
        # 清空已完成和已配送的作业
        self.completed_jobs = []
        self.dispatched_jobs = []

        # 初始化机器
        self.machines = self.case.machines.copy()
        
        # 初始化配送商
        self.distributors = self.case.distributors.copy()

        # 初始化动态到达事件
        self.arrival_events = []

        # 初始化统计指标
        self.total_weighted_tardiness = 0
        self.machine_utilization = []
        self.completion_times = {} # 工件的完成时间
        self.last_schedule_time = 0
        self.last_batch_time = 0

        return self._get_state()
    

    def _process_dynamic_arrivals(self):
        """处理动态作业到达"""
        if self.t == 0:  # t=0时不生成新作业
            return
        
        if len(self.available_jobs) == self.config.max_job_num_limit:
            # 如果当前作业数已达到限制，则不生成新作业
            return  
            
        # 基于概率生成新作业
        if np.random.random() < self.config.arrival_probability:
            # 泊松分布决定到达数量
            max_remaining_jobs = self.config.max_job_num_limit - len(self.available_jobs)
            num_arrivals = max(max_remaining_jobs, np.random.poisson(self.config.arrival_batch_size))
            for _ in range(num_arrivals):
                new_job = self.case._generate_one_job(len(self.available_jobs))
                self.available_jobs.append(new_job)
                # 记录到达事件
                self.arrival_events.append({
                    'time': self.t,
                    'job_id': new_job.job_id
                })

    def _log_state_transition(self, action: Dict, reward: float):
        """记录状态转换"""
        print(f"\n{'='*20}")
        print(f"时间步 {self.t}")
        
        
        # 1. 作业状态
        processing_jobs = [j for j in self.available_jobs if j.status == 'processing']
        print("作业状态:")
        print(f"- 可用作业数: {len(self.available_jobs)}")
        print(f"- 已加工作业数: {len(self.completed_jobs)}")
        print(f"- 已配送作业数: {len(self.dispatched_jobs)}")
        print(f"- 加工中作业数: {len(processing_jobs)}")
        
        # 2. 机器状态
        busy_machines = [m for m in self.machines if m.status == 'busy']
        print("\n机器状态:")
        print(f"- 总机器数: {len(self.machines)}")
        print(f"- 忙碌机器数: {len(busy_machines)}")
        print(f"- 机器利用率: {self.calculate_machine_utilization():.2%}")
        
        # 3. 动作执行详情
        print("\n执行的动作:")
        if 'wait' in action:
            print("- 执行等待")
        if 'schedule' in action:
            print("- 执行调度:")
            for job_id, machine_id in action['schedule'].items():
                print(f"  作业{job_id} -> 机器{machine_id}")
        if 'dispatch' in action:
            print("- 执行配送:")
            for batch_id, job_ids in action['dispatch'].items():
                print(f"  批次{batch_id}: 作业{job_ids}")
        
        # 4. 性能指标
        print("\n性能指标:")
        print(f"- 当前奖励: {reward:.2f}")
        print(f"- 总加权延迟: {self.total_weighted_tardiness:.2f}")
        print(f"- 作业推进比例: {self.calculate_operation_progress_ratio():.2%}")
        print(f"- 机器负载方差: {self.calculate_machine_load_variance():.2f}")
        
        # 5. 动态到达信息
        if self.arrival_events:
            print("\n本时间步新到达作业:")
            for event in self.arrival_events:
                if event['time'] == self.t:
                    print(f"- 作业{event['job_id']}")
                    
        

    def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
        """执行环境步进"""
        # 1. 时间步开始时的状态更新
        self._update_machine_states()     # 首先更新机器状态
        self._update_job_states()         # 更新作业状态
        self._update_dispatching_jobs()  # 更新配送状态
        
        
        # 2. 处理动态到达
        self._process_dynamic_arrivals()
        
        # 3. 执行决策动作
        if 'wait' in action:
            self._process_waiting(action['wait'])
        elif 'schedule' in action:        # 使用elif因为这些是互斥的决策
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
        return (self.t >= self.config.max_time_steps or 
                len(self.dispatched_jobs) == len(self.available_jobs))

    def _calculate_reward(self, action) -> float:
        reward = 0.0
        debug_info = {}
    
        # 基础动作奖励
        if 'wait' in action:
            reward -= 1.0  # 增加等待动作的惩罚
            debug_info['wait_penalty'] = -1.0
        elif 'schedule' in action:
            reward += 2.0  # 增加调度动作的奖励
            debug_info['schedule_base'] = 2.0
        elif 'dispatch' in action:
            reward += 2.0  # 增加配送动作的奖励
            debug_info['dispatch_base'] = 2.0
    
        # 调度质量奖励
        if 'schedule' in action:
            utilization = self.calculate_machine_utilization()
            reward += 5.0 * utilization
            debug_info['utilization'] = 5.0 * utilization
    
            job_progress = self.calculate_operation_progress_ratio()
            reward += 3.0 * job_progress  # 增加作业推进奖励
            debug_info['job_progress'] = 3.0 * job_progress
    
            load_balance = self.calculate_machine_load_variance()
            reward -= 0.5 * load_balance
            debug_info['load_balance_penalty'] = -0.5 * load_balance
    
        # 配送质量奖励
        if 'dispatch' in action:
            reward += 5.0  # 增加配送动作的奖励
            debug_info['dispatch_reward'] = 5.0

        # 计算配送完工时间延迟
        for job in self.completed_jobs:
            tardiness = max(0, job.dispatched_time - job.due_date)
            self.total_weighted_tardiness += tardiness
            reward -= tardiness 
            debug_info[f'job_{job.job_id}_tardiness'] = -tardiness

        
        # 计算分段配送时间要求延迟成本
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
                            reward -= int(penalty)
                            debug_info[f'distributor_{distributor.distributor_id}_due_time_{due_time}'] = -penalty

            
            
        # 即时操作奖励
        if self.operation_completed_this_step:
            reward += 1.0  # 增加工序完成奖励
            debug_info['operation_complete'] = 1.0
    
        if self.job_completed_this_step:
            reward += 2.0  # 增加作业完成奖励
            debug_info['job_complete'] = 2.0

        # 如果有作业被配送，增加奖励
        if self.job_dispatching_this_step:
            reward += 2.0  # 增加作业配送奖励
            debug_info['job_dispatching'] = 2.0

        if self.job_dispatched_this_step:
            reward += 3.0  # 增加作业配送奖励
            debug_info['job_dispatched'] = 3.0
    
        # 不限制奖励范围
        final_reward = reward
        debug_info['final_reward'] = final_reward
    
        print("Reward breakdown:", debug_info)
        return final_reward
    

    def calculate_machine_utilization(self) -> float:
        """计算机器利用率"""
        if self.t == 0:
            return 0.0
        total_busy_time = sum(1 for m in self.machines if m.status == 'busy')
        return total_busy_time / len(self.machines)

    def calculate_operation_progress_ratio(self):
        """计算作业生产推进进度比例（已完成工序数/总工序数）"""
        total_ops = sum(len(job.operations) for job in self.available_jobs)
        finished_ops = sum(job.current_operation + 1 for job in self.available_jobs)
        if total_ops == 0:
            return 0.0
        return finished_ops / total_ops

    def calculate_machine_load_variance(self):
        """计算机器负载方差（衡量负载均衡性，越小越均衡）"""
        loads = [m.remaining_time if m.status == 'busy' else 0 for m in self.machines]
        if not loads:
            return 0.0
        return float(np.var(loads))



    def _update_job_states(self):
        """更新作业状态"""
        for job in self.available_jobs:
            if job.status == 'completed':
                if job not in self.completed_jobs:
                    self.completed_jobs.append(job)
                    self.completion_times[job.job_id] = self.t
                    job.completed_time = self.t


    def _update_machine_states(self):
        """更新机器状态"""
        for machine in self.machines:
            if machine.status == 'busy':
                machine.remaining_time -= 1
                # 只在工序实际完成时更新状态
                if machine.remaining_time <= 0:
                    self._complete_operation(machine)

    def _complete_operation(self, machine: Machine):
        """完成一道工序"""
        job = next((j for j in self.available_jobs if j.job_id == machine.current_job), None)
        if job:
            # 更新工序进度
            job.current_operation += 1
            self.operation_completed_this_step = True
            
            # 检查是否所有工序都完成
            if job.current_operation >= len(job.operations)-1:
                job.status = 'completed'
                if job not in self.completed_jobs:
                    self.completed_jobs.append(job)
                    self.job_completed_this_step = True

            else:
                job.status = 'waiting'
                
            # 重置机器状态
            machine.status = 'waiting'
            machine.current_job = -1
            machine.remaining_time = 0


    def _complete_job(self, machine: Machine):
        """完成机器上的作业"""
        job = next((j for j in self.available_jobs if j.job_id == machine.current_job), None)
        if job:
            job.current_operation += 1
            if job.current_operation >= len(job.operations):
                job.status = 'completed'
            else:
                job.status = 'waiting'
            machine.status = 'waiting'
            machine.current_job = -1

    def _process_scheduling(self, schedule_action: Dict):
        """处理调度决策"""
        for job_id, machine_id in schedule_action.items():
            job = next((j for j in self.available_jobs if j.job_id == job_id), None)
            machine = next((m for m in self.machines if m.machine_id == machine_id), None)
            
            if job and machine and machine.status == 'waiting':
                # 获取当前工序
                current_op = job.operations[job.current_operation]
                if machine_id in current_op.available_machine_ids:
                    # 改变machine 状态
                    machine.status = 'busy'
                    machine.current_job = job.job_id
                    machine.remaining_time = current_op.processing_times[machine_id]
                
                    # 改变job 状态
                    # 更新作业状态
                    job.status = 'processing'
                    self.machine_utilization.append(1.0)


    def _process_dispatching(self, dispatch_action: Dict):
        """处理配送决策"""
        BASE_DELIVERY_TIME = 1  # 基础配送时间
        PER_JOB_TIME = 0        # 每多一个工件增加的配送时间
        for batch_id, job_ids in dispatch_action.items():
            # 只处理已完成加工的作
            flat_job_ids = list(chain.from_iterable(job_ids)) if job_ids and isinstance(job_ids[0], list) else job_ids

            dispatch_jobs = [j for j in self.completed_jobs 
                            if j.job_id in flat_job_ids]
           
            if dispatch_jobs:
                # 创建新批次,设置配送时间
                batch_size = len(dispatch_jobs)
                delivery_time = BASE_DELIVERY_TIME + PER_JOB_TIME * batch_size

                self.batches.append({
                    'batch_id': batch_id,
                    'jobs': dispatch_jobs,
                    'dispatch_start_time': self.t,
                    'status': 'dispatching'
                })
                
                # 更新作业状态
                for job in dispatch_jobs:
                    job.status = 'dispatching'  
                    job.dispatch_remaining_time = delivery_time  # 设置配送时间
                    self.job_dispatching_this_step = True

    def _update_dispatching_jobs(self):
        """推进所有配送中的作业，配送完成后更新状态和时间"""
        for job in self.completed_jobs:
            if getattr(job, 'status', None) == 'dispatching':
                job.dispatch_remaining_time -= 1
                if job.dispatch_remaining_time <= 0:
                    job.status = 'dispatched'
                    job.dispatch_time = self.t  # 配送完成时间
                    if job not in self.dispatched_jobs:
                        self.dispatched_jobs.append(job)
                        self.job_dispatched_this_step = True

    def _process_waiting(self, wait_action):
        """
        处理等待决策。
        Args:
            wait_action: 等待动作的参数（可根据实际需求扩展）
        """
        # 这里可以根据需要实现等待逻辑，例如什么都不做，仅占用时间步
        pass
        
        

               
    def _get_state(self) -> Dict:
        """获取当前状态"""
        return {
            'current_time': self.t,
            'last_schedule_time': self.last_schedule_time,
            'last_batch_time': self.last_batch_time,
            'available_jobs': self.available_jobs,
            'machines': self.machines,
            'distributors': self.distributors,
            'completed_jobs': self.completed_jobs,
            'dispatched_jobs': self.dispatched_jobs
        }
