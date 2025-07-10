from config import Config
from typing import Dict, Tuple
from data_structures import Job, Operation,Machine,Distributor,DeliveryRequirement
from case_generator import FlexibleJobShopScenario
import numpy as np


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
        
        # 执行重置
        self.reset()


    def reset(self) -> Dict:
        self.t = 0
        self.done = False

        # 初始化作业
        self.initial_jobs = self.case.jobs.copy()
        # 包括新到达的作业
        self.available_jobs = self.case.jobs.copy()
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
            
        # 基于概率生成新作业
        if np.random.random() < self.config.arrival_probability:
            # 泊松分布决定到达数量
            num_arrivals = np.random.poisson(self.config.arrival_batch_size)
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
        """
        计算奖励函数，综合考虑等待、调度和配送三类动作的效果。
        """
        reward = 0.0

        if 'wait' in action:
            # 等待惩罚，鼓励主动调度
            reward -= 1.0

        if 'schedule' in action:
            # 奖励高资源利用率、作业推进和负载均衡
            utilization = self.calculate_machine_utilization()
            job_progress = self.calculate_operation_progress_ratio()
            load_balance = self.calculate_machine_load_variance()

            reward += 2.0 * utilization
            reward += 1.0 * job_progress
            reward -= 1.0 * load_balance

        if 'dispatch' in action:
            # 奖励及时配送，惩罚延迟和未满足配送需求
            dispatched_jobs = self.dispatched_jobs
            distributor_map = {}
            for job in dispatched_jobs:
                distributor_map.setdefault(job.distributor_id, []).append(job)

            for distributor in self.distributors:
                jobs = distributor_map.get(distributor.distributor_id, [])
                delivery_req = distributor.delivery_requirements
                total_jobs = len(distributor.assigned_jobs)

                for due_time, ratio, weight in zip(
                    delivery_req.due_times,
                    delivery_req.ratios,
                    delivery_req.weights
                ):
                    # 满足截止时间的作业数
                    completed = sum(
                        1 for j in jobs
                        if getattr(j, 'dispatch_time', None) is not None and j.dispatch_time <= due_time
                    )
                    required = int(np.ceil(ratio * total_jobs))
                    tardy = max(0, required - completed)
                    reward -= weight * tardy
                    if tardy == 0 and required > 0:
                        reward += 10.0

                # 最终截止时间惩罚
                if total_jobs > 0 and len(jobs) == total_jobs:
                    latest_dispatch = max(getattr(j, 'dispatch_time', 0) for j in jobs)
                    final_due = max(delivery_req.due_times)
                    tardiness_time = max(0, latest_dispatch - final_due)
                    reward -= 5.0 * tardiness_time
                    if tardiness_time == 0:
                        reward += 5.0

        return float(reward)

    def calculate_machine_utilization(self) -> float:
        """计算机器利用率"""
        if self.t == 0:
            return 0.0
        total_busy_time = sum(1 for m in self.machines if m.status == 'busy')
        return total_busy_time / len(self.machines)

    def calculate_operation_progress_ratio(self):
        """计算作业推进进度比例（已完成工序数/总工序数）"""
        total_ops = sum(len(job.operations) for job in self.available_jobs)
        finished_ops = 1+sum(job.current_operation for job in self.available_jobs)
        if total_ops == 0:
            return 0.0
        return finished_ops / total_ops

    def calculate_machine_load_variance(self):
        """计算机器负载方差（衡量负载均衡性，越小越均衡）"""
        loads = [m.remaining_time if m.status == 'busy' else 0 for m in self.machines]
        if not loads:
            return 0.0
        return np.var(loads)



    def _update_job_states(self):
        """更新作业状态"""
        for job in self.available_jobs:
            if job.status == 'completed':
                if job not in self.completed_jobs:
                    self.completed_jobs.append(job)
                    self.completion_times[job.job_id] = self.t


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
            
            # 检查是否所有工序都完成
            if job.current_operation >= len(job.operations)-1:
                job.status = 'completed'
                if job not in self.completed_jobs:
                    self.completed_jobs.append(job)
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
        for batch_id, job_ids in dispatch_action.items():
            # 只处理已完成加工的作业
            dispatch_jobs = [j for j in self.completed_jobs 
                            if j.job_id in job_ids and j.status == 'completed']
            self.dispatched_jobs.extend(dispatch_jobs)
            if dispatch_jobs:
                # 创建新批次,设置配送时间
                self.batches.append({
                    'batch_id': batch_id,
                    'jobs': dispatch_jobs,
                    'dispatch_time': self.t,
                    'status': 'dispatched'
                })
                
                # 更新作业状态
                for job in dispatch_jobs:
                    job.status = 'dispatched'  


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