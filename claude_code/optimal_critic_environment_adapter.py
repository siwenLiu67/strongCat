"""
Optimal-Critic环境适配器 - 使Option-Critic算法能够复用WarehouseEnvironment
参考PPOEnvironmentAdapter的结构，但保持Option-Critic算法的特性
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any
from environment import WarehouseEnvironment
from case_generator import FlexibleJobShopScenario
from config import Config
from data_structures import Job, Operation, Machine, Distributor


class OptimalCriticEnvironmentAdapter:
    """Optimal-Critic环境适配器，将WarehouseEnvironment适配为Option-Critic可用的接口"""
    
    def __init__(self, config: Config, scenario: FlexibleJobShopScenario):
        """初始化适配器
        
        Args:
            config: 配置对象
            scenario: 算例生成器实例
        """
        self.base_env = WarehouseEnvironment(config, scenario)
        self.scenario = scenario
        self.config = config
        
        # Option-Critic特有的状态和动作空间参数
        self.state_dim = self._calculate_state_dim()
        self.action_dim = self._calculate_action_dim()
        
        # 选项语义定义（保持Option-Critic特性）
        self.option_semantics = {
            0: "urgent_jobs_first",      # 紧急作业优先
            1: "shortest_job_first",     # 短作业优先  
            2: "load_balancing",         # 负载均衡
            3: "machine_utilization",    # 机器利用率优化
            4: "batch_optimization",     # 批次优化
            5: "early_dispatch",         # 早期派遣
            6: "quality_focus",          # 质量关注
            7: "exploration_mode"        # 探索模式
        }
        
        # 选项上下文（保持Option-Critic特性）
        self.option_context = {
            'current_focus': 'schedule',  # schedule or dispatch
            'urgency_level': 0.0,
            'machine_utilization': 0.0,
            'dispatch_pressure': 0.0
        }
        
        # 性能指标跟踪
        self.tardiness_penalty = 0
        self.makespan_penalty = 0
        self.dispatch_penalty = 0
        
        self.reset()
    
    def _calculate_state_dim(self) -> int:
        """计算状态空间维度（保持Option-Critic的特性）"""
        # 与原始Option-Critic保持一致的状态维度
        job_features = len(self.scenario.jobs) * 6  # 增加更多作业特征
        machine_features = len(self.scenario.machines) * 4  # 增加机器特征
        time_features = 5  # 增加时间特征
        dispatch_features = len(self.scenario.distributors) * 3  # 增加派遣特征
        global_features = 8  # 全局特征
        
        return job_features + machine_features + time_features + dispatch_features + global_features
    
    def _calculate_action_dim(self) -> int:
        """计算动作空间维度（保持Option-Critic的特性）"""
        # 与原始Option-Critic保持一致的动作维度
        schedule_actions = 4  # 作业排序、机器选择、优先级、时机
        dispatch_actions = 3  # 批次大小、派遣时机、配送商选择
        
        return schedule_actions + dispatch_actions
    
    def reset(self) -> np.ndarray:
        """重置环境并返回初始状态"""
        self.base_env.reset()
        self.tardiness_penalty = 0
        self.makespan_penalty = 0
        self.dispatch_penalty = 0
        
        # 重置选项上下文
        self.option_context = {
            'current_focus': 'schedule',
            'urgency_level': 0.0,
            'machine_utilization': 0.0,
            'dispatch_pressure': 0.0
        }
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """将WarehouseEnvironment的状态转换为Option-Critic需要的numpy数组状态"""
        state = []
        
        # 获取基础环境状态
        base_state = self.base_env._get_state()
        jobs = base_state['available_jobs'] + base_state['completed_jobs']
        machines = base_state['machines']
        distributors = base_state['distributors']
        
        # 作业状态特征
        for job in self.scenario.jobs:
            if job.job_id in [j.job_id for j in base_state['completed_jobs']]:
                state.extend([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])  # 已完成
            else:
                # 查找当前作业状态
                current_job = next((j for j in jobs if j.job_id == job.job_id), None)
                if current_job:
                    progress = current_job.current_operation / len(current_job.operations)
                    remaining_ops = len(current_job.operations) - current_job.current_operation
                    
                    # 计算平均处理时间
                    avg_proc_time = np.mean([min(op.processing_times.values()) 
                                           for op in current_job.operations]) if current_job.operations else 0
                    remaining_time = remaining_ops * avg_proc_time / 100.0
                    
                    due_date = getattr(current_job, 'due_date', 1000)
                    urgency = max(0, due_date - base_state['current_time']) / 100.0
                    tardiness = max(0, base_state['current_time'] - due_date) / 100.0
                    
                    all_due_dates = [getattr(j, 'due_date', 1000) for j in jobs]
                    relative_priority = (due_date - min(all_due_dates)) / max(max(all_due_dates) - min(all_due_dates), 1)
                    
                    state.extend([progress, remaining_time, urgency, tardiness, 0.0, relative_priority])
                else:
                    state.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 作业不存在
        
        # 机器状态特征
        for machine in self.scenario.machines:
            current_machine = next((m for m in machines if m.machine_id == machine.machine_id), None)
            if current_machine:
                busy = 1.0 if current_machine.status == 'busy' else 0.0
                
                # 计算机器负载
                scheduled_ops = [j for j in jobs 
                               if hasattr(j, 'status') and j.status == 'processing' 
                               and hasattr(j, 'current_machine') and j.current_machine == machine.machine_id]
                load = len(scheduled_ops) / max(len(jobs), 1)
                
                available_time = current_machine.remaining_time / 100.0 if current_machine.status == 'busy' else 0.0
                
                # 计算效率（简化）
                efficiency = 1.0 / (available_time + 1) if current_machine.status == 'busy' else 0.5
                
                state.extend([busy, load, available_time, efficiency])
            else:
                state.extend([0.0, 0.0, 0.0, 0.5])
        
        # 时间特征
        time_progress = min(base_state['current_time'] / 200.0, 1.0)
        
        # 计算平均延迟
        avg_tardiness = np.mean([max(0, base_state['current_time'] - getattr(job, 'due_date', 1000)) 
                               for job in jobs]) / 100.0
        
        completion_rate = len(base_state['completed_jobs']) / max(len(jobs), 1)
        
        # 计算机器负载方差
        machine_busy_times = [m.remaining_time for m in machines if m.status == 'busy']
        workload_variance = np.var(machine_busy_times) / 10000.0 if machine_busy_times else 0.0
        
        total_operations = sum(len(job.operations) for job in jobs)
        scheduled_operations_count = sum(1 for job in jobs if job.status == 'processing')
        schedule_progress = scheduled_operations_count / max(total_operations, 1)
        
        state.extend([time_progress, avg_tardiness, completion_rate, workload_variance, schedule_progress])
        
        # 派遣状态特征
        for distributor in self.scenario.distributors:
            dist_jobs = [job for job in jobs 
                        if getattr(job, 'distributor_id', 0) == distributor.distributor_id]
            completed_dist_jobs = [job for job in dist_jobs if job in base_state['completed_jobs']]
            
            batch_count = len([b for b in getattr(self.base_env, 'batches', []) 
                              if b.get('status') == 'dispatching'])
            waiting_jobs = len(completed_dist_jobs)
            
            if batch_count > 0:
                avg_batch_size = waiting_jobs / batch_count
                dispatch_efficiency = min(avg_batch_size / 3.0, 1.0)
            else:
                dispatch_efficiency = 0.0
            
            state.extend([batch_count / 10.0, waiting_jobs / max(len(dist_jobs), 1), dispatch_efficiency])
        
        # 全局特征
        global_features = [
            self.option_context['urgency_level'],
            self.option_context['machine_utilization'],
            self.option_context['dispatch_pressure'],
            len([j for j in jobs if j.status == 'waiting']) / max(len(jobs), 1),
            base_state['current_time'] / 1000.0,
            min(self._get_total_reward() / 100.0, 1.0),
            len(getattr(self.base_env, 'batches', [])) / 10.0,
            (self.tardiness_penalty + self.makespan_penalty + self.dispatch_penalty) / 300.0
        ]
        
        state.extend(global_features)
        
        # 确保状态维度正确
        if len(state) < self.state_dim:
            state.extend([0.0] * (self.state_dim - len(state)))
        elif len(state) > self.state_dim:
            state = state[:self.state_dim]
        
        return np.array(state, dtype=np.float32)
    
    def _get_total_reward(self) -> float:
        """计算总奖励（简化版本）"""
        # 这里可以根据需要实现更复杂的奖励计算
        return - (self.tardiness_penalty + self.makespan_penalty + self.dispatch_penalty)
    
    def _decode_action(self, action: np.ndarray, option: int) -> Dict:
        """将Option-Critic的连续动作解码为WarehouseEnvironment的字典动作"""
        action = np.clip(action, -1, 1)
        
        # 根据选项调整动作解释（保持Option-Critic特性）
        interpreted_action = self._interpret_action_with_option(action, option)
        
        schedule_action = interpreted_action[:4]
        dispatch_action = interpreted_action[4:7]
        
        env_action = {'wait': {}}
        
        # 调度动作解码
        if schedule_action[0] > 0 and self._has_schedulable_jobs():
            job_id, machine_id = self._select_schedule_targets(schedule_action, option)
            if job_id is not None and machine_id is not None:
                env_action = {'schedule': {job_id: machine_id}}
        
        # 派遣动作解码
        elif dispatch_action[1] > 0 and self._has_dispatchable_jobs():
            distributor_id, job_ids = self._select_dispatch_targets(dispatch_action, option)
            if distributor_id is not None and job_ids:
                env_action = {'dispatch': {distributor_id: job_ids}}
        
        return env_action
    
    def _interpret_action_with_option(self, action: np.ndarray, option: int) -> np.ndarray:
        """根据选项语义调整动作解释（保持Option-Critic特性）"""
        interpreted_action = action.copy()
        
        if option == 0:  # urgent_jobs_first
            interpreted_action[0] = abs(interpreted_action[0])
        elif option == 1:  # shortest_job_first
            interpreted_action[1] = -abs(interpreted_action[1])
        elif option == 2:  # load_balancing
            interpreted_action[2] = 0.0
        elif option == 3:  # machine_utilization
            interpreted_action[3] = abs(interpreted_action[3])
        elif option == 4:  # batch_optimization
            interpreted_action[4] = abs(interpreted_action[4])
        elif option == 5:  # early_dispatch
            interpreted_action[5] = abs(interpreted_action[5])
        elif option == 6:  # quality_focus
            interpreted_action[1] = abs(interpreted_action[1])
        elif option == 7:  # exploration_mode
            noise = np.random.normal(0, 0.2, size=interpreted_action.shape)
            interpreted_action = np.clip(interpreted_action + noise, -1, 1)
        
        return interpreted_action
    
    def _has_schedulable_jobs(self) -> bool:
        """检查是否有可调度的作业"""
        base_state = self.base_env._get_state()
        waiting_jobs = [j for j in base_state['available_jobs'] if j.status == 'waiting']
        idle_machines = [m for m in base_state['machines'] if m.status == 'waiting']
        return len(waiting_jobs) > 0 and len(idle_machines) > 0
    
    def _has_dispatchable_jobs(self) -> bool:
        """检查是否有可派遣的作业"""
        base_state = self.base_env._get_state()
        completed_not_dispatched = [j for j in base_state['completed_jobs'] 
                                  if j not in base_state['dispatched_jobs']]
        return len(completed_not_dispatched) > 0
    
    def _select_schedule_targets(self, schedule_action: np.ndarray, option: int) -> Tuple[Optional[int], Optional[int]]:
        """选择调度目标和机器"""
        base_state = self.base_env._get_state()
        waiting_jobs = [j for j in base_state['available_jobs'] if j.status == 'waiting']
        idle_machines = [m for m in base_state['machines'] if m.status == 'waiting']
        
        if not waiting_jobs or not idle_machines:
            return None, None
        
        # 根据选项计算作业优先级
        job_priorities = []
        for job in waiting_jobs:
            due_date = getattr(job, 'due_date', 1000)
            progress = job.current_operation / len(job.operations)
            
            if option == 0:  # urgent_jobs_first
                priority = 1000 - max(0, due_date - base_state['current_time'])
            elif option == 1:  # shortest_job_first
                current_op = job.operations[job.current_operation]
                avg_proc_time = np.mean(list(current_op.processing_times.values())) if current_op.processing_times else 0
                priority = -avg_proc_time
            elif option == 6:  # quality_focus
                priority = len(current_op.processing_times) if job.current_operation < len(job.operations) else 0
            else:
                urgency = max(0, due_date - base_state['current_time'])
                priority = urgency * (1 - schedule_action[2]) + progress * schedule_action[2]
            
            job_priorities.append((job.job_id, priority))
        
        # 选择作业
        if schedule_action[0] > 0:
            selected_job_id = max(job_priorities, key=lambda x: x[1])[0]
        else:
            selected_job_id = random.choice(job_priorities)[0] if job_priorities else None
        
        if selected_job_id is None:
            return None, None
        
        # 选择机器
        selected_job = next(j for j in waiting_jobs if j.job_id == selected_job_id)
        current_op = selected_job.operations[selected_job.current_operation]
        eligible_machines = [m for m in idle_machines if m.machine_id in current_op.available_machine_ids]
        
        if not eligible_machines:
            return None, None
        
        if schedule_action[1] > 0:
            # 选择处理时间最短的机器
            machine_scores = []
            for machine in eligible_machines:
                proc_time = current_op.processing_times.get(machine.machine_id, float('inf'))
                machine_scores.append((machine.machine_id, -proc_time))
            selected_machine_id = max(machine_scores, key=lambda x: x[1])[0]
        else:
            selected_machine_id = random.choice(eligible_machines).machine_id
        
        return selected_job_id, selected_machine_id
    
    def _select_dispatch_targets(self, dispatch_action: np.ndarray, option: int) -> Tuple[Optional[int], List[int]]:
        """选择派遣目标和作业"""
        base_state = self.base_env._get_state()
        completed_not_dispatched = [j for j in base_state['completed_jobs'] 
                                  if j not in base_state['dispatched_jobs']]
        
        if not completed_not_dispatched:
            return None, []
        
        # 按配送商分组
        distributor_jobs = {}
        for job in completed_not_dispatched:
            dist_id = getattr(job, 'distributor_id', 0)
            if dist_id not in distributor_jobs:
                distributor_jobs[dist_id] = []
            distributor_jobs[dist_id].append(job)
        
        # 选择配送商
        if not distributor_jobs:
            return None, []
        
        # 根据选项选择配送商
        if option == 5:  # early_dispatch
            # 选择作业最多的配送商
            selected_distributor = max(distributor_jobs.items(), key=lambda x: len(x[1]))[0]
        elif option == 4:  # batch_optimization
            # 选择作业数量适中的配送商（3-5个作业）
            suitable_distributors = [(dist_id, jobs) for dist_id, jobs in distributor_jobs.items() 
                                   if 3 <= len(jobs) <= 5]
            if suitable_distributors:
                selected_distributor = random.choice(suitable_distributors)[0]
            else:
                selected_distributor = random.choice(list(distributor_jobs.keys()))
        else:
            selected_distributor = random.choice(list(distributor_jobs.keys()))
        
        # 选择作业
        jobs_to_dispatch = distributor_jobs[selected_distributor]
        
        # 根据选项选择作业数量
        if option == 4:  # batch_optimization
            # 批量优化：选择所有作业
            job_ids = [job.job_id for job in jobs_to_dispatch]
        elif option == 5:  # early_dispatch
            # 早期派遣：选择前几个作业
            max_jobs = min(3, len(jobs_to_dispatch))
            job_ids = [job.job_id for job in jobs_to_dispatch[:max_jobs]]
        else:
            # 默认：随机选择1-3个作业
            num_jobs = min(max(1, int(abs(dispatch_action[0]) * 3)), len(jobs_to_dispatch))
            job_ids = [job.job_id for job in random.sample(jobs_to_dispatch, num_jobs)]
        
        return selected_distributor, job_ids
    
    def step(self, action: np.ndarray, option: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行环境步进（Option-Critic接口）"""
        # 解码动作
        env_action = self._decode_action(action, option)
        
        # 执行基础环境步进
        base_state, reward, done, info = self.base_env.step(env_action)
        
        # 更新选项上下文（保持Option-Critic特性）
        self._update_option_context(env_action, reward)
        
        # 更新性能指标
        self._update_performance_metrics()
        
        # 获取新状态
        new_state = self._get_state()
        
        # 计算Option-Critic特有的内在奖励
        intrinsic_reward = self._calculate_intrinsic_reward(option, env_action)
        total_reward = reward + intrinsic_reward
        
        return new_state, total_reward, done, info
    
    def _update_option_context(self, action: Dict, reward: float):
        """更新选项上下文（保持Option-Critic特性）"""
        base_state = self.base_env._get_state()
        
        # 更新紧急程度
        waiting_jobs = [j for j in base_state['available_jobs'] if j.status == 'waiting']
        if waiting_jobs:
            avg_urgency = np.mean([max(0, getattr(j, 'due_date', 1000) - base_state['current_time']) 
                                 for j in waiting_jobs]) / 100.0
            self.option_context['urgency_level'] = min(avg_urgency, 1.0)
        
        # 更新机器利用率
        self.option_context['machine_utilization'] = self.base_env.calculate_machine_utilization()
        
        # 更新派遣压力
        completed_not_dispatched = len([j for j in base_state['completed_jobs'] 
                                      if j not in base_state['dispatched_jobs']])
        self.option_context['dispatch_pressure'] = min(completed_not_dispatched / 10.0, 1.0)
        
        # 更新当前焦点
        if 'schedule' in action:
            self.option_context['current_focus'] = 'schedule'
        elif 'dispatch' in action:
            self.option_context['current_focus'] = 'dispatch'
    
    def _update_performance_metrics(self):
        """更新性能指标"""
        base_state = self.base_env._get_state()
        
        # 计算延迟惩罚
        for job in base_state['completed_jobs']:
            tardiness = max(0, getattr(job, 'dispatched_time', base_state['current_time']) - 
                          getattr(job, 'due_date', 1000))
            self.tardiness_penalty += tardiness
        
        # 计算制造周期惩罚
        if base_state['dispatched_jobs']:
            max_completion = max([getattr(job, 'dispatched_time', 0) for job in base_state['dispatched_jobs']])
            self.makespan_penalty += max_completion
        
        # 计算派遣惩罚
        for distributor in base_state['distributors']:
            requirement = getattr(distributor, 'delivery_requirements', None)
            if requirement:
                for due_time, ratio, weight in zip(requirement.due_times, requirement.ratios, requirement.weights):
                    completed_jobs = [j for j in base_state['completed_jobs'] 
                                    if getattr(j, 'dispatched_time', float('inf')) <= due_time]
                    completed_amount = sum(getattr(j, 'amount', 0) for j in completed_jobs)
                    required_amount = ratio * getattr(distributor, 'total_amount', 0)
                    if completed_amount < required_amount:
                        penalty = (required_amount - completed_amount) * weight
                        self.dispatch_penalty += penalty
    
    def _calculate_intrinsic_reward(self, option: int, action: Dict) -> float:
        """计算Option-Critic特有的内在奖励"""
        intrinsic_reward = 0.0
        
        # 选项一致性奖励
        if option == 0 and 'schedule' in action:  # urgent_jobs_first + 调度
            intrinsic_reward += 0.1
        elif option == 1 and 'schedule' in action:  # shortest_job_first + 调度
            intrinsic_reward += 0.1
        elif option == 4 and 'dispatch' in action:  # batch_optimization + 派遣
            intrinsic_reward += 0.2
        elif option == 5 and 'dispatch' in action:  # early_dispatch + 派遣
            intrinsic_reward += 0.1
        
        # 选项切换惩罚
        if (self.option_context['current_focus'] == 'schedule' and 'dispatch' in action) or \
           (self.option_context['current_focus'] == 'dispatch' and 'schedule' in action):
            intrinsic_reward -= 0.05
        
        return intrinsic_reward
    
    def get_state_dim(self) -> int:
        """获取状态空间维度"""
        return self.state_dim
    
    def get_action_dim(self) -> int:
        """获取动作空间维度"""
        return self.action_dim
    
    def get_option_semantics(self) -> Dict[int, str]:
        """获取选项语义"""
        return self.option_semantics
    
    def get_option_context(self) -> Dict[str, float]:
        """获取选项上下文"""
        return self.option_context
