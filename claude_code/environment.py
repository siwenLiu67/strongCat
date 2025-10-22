from config import Config
from typing import Dict, Tuple
from data_structures import Job, Operation,Machine,Distributor,DeliveryRequirement
from case_generator import FlexibleJobShopScenario
import numpy as np
from itertools import chain
class RunningStat:
    def __init__(self, eps=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps

    def update(self, x):
        # Welford online update
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var = ((self.count - 1) * self.var + delta * delta2) / self.count

    def normalize(self, x):
        std = max(1e-6, self.var ** 0.5)
        return (x - self.mean) / std



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
        # 在 env.__init__ 中
        self._reward_stat = RunningStat()
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
        self.next_job_id = len(case.jobs)  # 初始工件数
        
        # 初始化机器和配送商
        self.machines = []
        self.distributors = []
        self.batches=[]
        self.this_step_complete_jobs = 0
        self.this_step_dispatch_jobs = 0
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
        self.remaining_dynamic_jobs = config.num_dynamic_jobs  # 剩余要到达的动态作业数
        self.dynamic_jobs_arrived = 0  # 已到达的动态作业数
        self.arrival_events = []  # 到达事件记录
    
        
        # 执行重置
        self.reset()


    def reset(self) -> Dict:
        """重置环境"""
        self.t = 0
        self.tardy_penalty = 0
        self.done = False

        # 初始化作业 - 只包含初始工件
        self.initial_jobs = self.case.jobs.copy()
        self.available_jobs = self.case.jobs.copy()  # 开始时只有初始工件
        self.this_step_complete_jobs = 0
        self.this_step_dispatch_jobs = 0
        # 重设作业状态
        for job in self.available_jobs:
            job.status = 'waiting'
            job.current_operation = 0
            job.job_type = 'initial'  # 标记为初始工件
        
        # 清空已完成和已配送的作业
        self.completed_jobs = []
        self.dispatched_jobs = []

        # 初始化机器和配送商
        self.machines = self.case.machines.copy()
        self.distributors = self.case.distributors.copy()

        # 重置动态到达计数器
        self.remaining_dynamic_jobs = self.config.num_dynamic_jobs
        self.dynamic_jobs_arrived = 0
        self.arrival_events = []

        # 初始化统计指标
        self.total_weighted_tardiness = 0
        self.machine_utilization = 0
        self.completion_times = {}
        self.last_schedule_time = 0
        self.last_batch_time = 0

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
        arrival_probability = getattr(self.config, 'batch_arrival_probability', 0.75)  # 每个时间步75%概率到达
        
        if np.random.random() < arrival_probability:
            # 随机决定本次到达的作业数量
            min_batch_size = getattr(self.config, 'min_batch_size', 1)
            max_batch_size = min(
                self.remaining_dynamic_jobs,
                getattr(self.config, 'max_batch_size', 5)  # 每次最多到达5个
            )
            
            if max_batch_size >= min_batch_size:
                # 随机生成批量大小
                batch_size = np.random.randint(min_batch_size, max_batch_size + 1)
                
                arrived_jobs = []
                for i in range(batch_size):
                    if self.remaining_dynamic_jobs > 0:
                    
                        new_job_id = self.next_job_id
                        
                        # 生成新作业
                        new_job = self.case._generate_one_job(new_job_id)
                        new_job.arrival_time = self.t  # 设置到达时间
                        new_job.job_type = 'dynamic'   # 标记为动态工件
                        new_job.status = 'waiting'     # 初始状态为等待
                        new_job.current_operation = 0  # 从第一道工序开始

                        self._update_distributor_job_mapping()

                        
                        # 添加到可用作业列表
                        self.available_jobs.append(new_job)
                        arrived_jobs.append(new_job)
                        
                        self.next_job_id += 1  # 保证每次唯一
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
        reward = self._calculate_reward(action, episode_progress=self.t / self.config.max_time_steps)
        
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


    def _calculate_reward(self, action, episode_progress) -> float:
        """
        最终优化的环境奖励函数
        - 明确的渐进式引导
        - 避免过大惩罚导致躺平  
        - 合理的奖励归一化
        - 自适应训练阶段调整
        episode_progress: 当前训练进度 [0,1]
        """
        total_reward = 0.0
        debug_info = {}
        
        # === 阶段自适应参数 ===
        if episode_progress < 0.3:  # 前期：鼓励探索
            penalty_weight = 0.3
            exploration_bonus = 0.05
            milestone_threshold = 2  # 较低的里程碑门槛
        elif episode_progress < 0.7:  # 中期：平衡学习
            penalty_weight = 0.7
            exploration_bonus = 0.02
            milestone_threshold = 3
        else:  # 后期：严格优化
            penalty_weight = 1.0
            exploration_bonus = 0.0
            milestone_threshold = 4
        
        total_reward += exploration_bonus
        debug_info['exploration_bonus'] = exploration_bonus
        
        # === 核心正向奖励（明确的好的策略）===
        
        # 1. 作业完成奖励 - 最明确的成功信号    
        new_completions =  self.this_step_complete_jobs
        
        if new_completions > 0:
            completion_reward = new_completions * 0.8  # 明确但合理的奖励
            total_reward += completion_reward
            debug_info['completion_reward'] = completion_reward
        
        # 2. 配送效率奖励 - 鼓励及时行动
        new_dispatches = self.this_step_dispatch_jobs
        
        if new_dispatches > 0:
            dispatch_reward = new_dispatches * 0.4
            total_reward += dispatch_reward
            debug_info['dispatch_reward'] = dispatch_reward
        
        # 3. 里程碑奖励 - 防止后期动力不足
        milestone_bonus = self._check_milestone(milestone_threshold)
        total_reward += milestone_bonus
        if milestone_bonus > 0:
            debug_info['milestone_bonus'] = milestone_bonus
        
        # 4. 资源高效利用奖励
        utilization = self.calculate_machine_utilization()
        if 0.6 <= utilization <= 0.85:  # 最佳利用率区间
            util_reward = 0.3
            total_reward += util_reward
            debug_info['util_reward'] = util_reward
        elif utilization > 0.9:  # 轻微过载
            util_penalty = 0.1 * penalty_weight
            total_reward -= util_penalty
            debug_info['util_penalty'] = util_penalty
        
        # 5. 积极决策奖励 - 鼓励采取行动
        if self._is_action_effective(action):
            action_reward = 0.15
            total_reward += action_reward
            debug_info['action_reward'] = action_reward
        
        # === 温和的负向惩罚（避免躺平）===
        
        # 1. 延误惩罚 - 重要但不过度
        tardiness_penalty = self._calculate_tardiness_penalty()
        if tardiness_penalty > 0:
            # 使用渐进式惩罚：延误越大，惩罚增长越缓
            adjusted_penalty = min(tardiness_penalty, 1.0) * penalty_weight * 0.5
            total_reward -= adjusted_penalty
            debug_info['tardiness_penalty'] = adjusted_penalty
        
        # 2. 资源闲置惩罚 - 温和提醒
        idle_penalty = self._calculate_idle_penalty()
        if idle_penalty > 0:
            adjusted_penalty = idle_penalty * penalty_weight * 0.3
            total_reward -= adjusted_penalty
            debug_info['idle_penalty'] = adjusted_penalty
        
        # 3. 系统停滞惩罚 - 避免死锁
        if not self._is_system_active():
            stagnant_penalty = 0.1 * penalty_weight
            total_reward -= stagnant_penalty
            debug_info['stagnant_penalty'] = stagnant_penalty
        
        # === 基础保障奖励 ===
        base_reward = 0.1  # 很小的基础奖励，避免过度负奖励
        total_reward += base_reward
        debug_info['base_reward'] = base_reward
        
        # === 最终归一化和范围控制 ===
        # 严格控制在合理范围内
        normalized_reward = float(np.clip(total_reward, -1.0, 3.0))
    
        
        debug_info['final_reward'] = normalized_reward
        debug_info['raw_reward'] = total_reward
        debug_info['penalty_weight'] = penalty_weight
        
    
        return normalized_reward

    def _check_milestone(self, threshold):
        """检查里程碑奖励"""
        current_completed = len(self.completed_jobs)
      
        if current_completed / len(self.available_jobs) >= 0.5: 
            return 1.5  # 合理的里程碑奖励
        return 0.0

    def _calculate_tardiness_penalty(self):
        """计算延误惩罚（归一化）"""
        completed_jobs = self.completed_jobs
        if not completed_jobs:
            return 0.0
        
        total_tardiness = sum(max(0, getattr(job, 'dispatched_time', self.t) - job.due_date) 
                            for job in completed_jobs)
        
        # 归一化：平均延误时间 / 最大容忍延误
        avg_tardiness = total_tardiness / len(completed_jobs)
        max_tolerable_tardiness = 50  # 假设最大容忍50时间单位
        
        return min(avg_tardiness / max_tolerable_tardiness, 1.0)

    def _calculate_idle_penalty(self):
        """计算资源闲置惩罚"""
        idle_machines = len([m for m in self.machines if m.status == 'waiting'])
        waiting_jobs = len([j for j in self.available_jobs if j.status == 'waiting'])
        
        if idle_machines > 0 and waiting_jobs > 0:
            # 有闲置资源且有作业等待的惩罚
            return min(idle_machines / len(self.machines), 0.5)
        return 0.0

    def _is_action_effective(self, action):
        """判断动作是否有效"""
        if 'schedule' in action and action['schedule']:
            return True
        if 'dispatch' in action and action['dispatch']:
            return True
        return False

    def _is_system_active(self):
        """判断系统是否活跃"""
        has_schedulable = any(job.status == 'waiting' for job in self.available_jobs)
        has_idle_machines = any(machine.status == 'waiting' for machine in self.machines)
        has_dispatchable = len(self.completed_jobs) > len(self.dispatched_jobs)
        
        return has_schedulable or has_dispatchable

    def _get_episode_progress(self, episode, total_episodes):
        """计算当前训练进度 [0,1]"""
        return min(1.0, episode / total_episodes)

    # def _calculate_reward(self, action) -> float:
    #     reward = 0.0
    #     debug_info = {}
        
    #     total_jobs = len(self.completed_jobs) + len(self.available_jobs) + len(self.dispatched_jobs)
    #     if total_jobs == 0:
    #         return 0.0
        
    #     # === 增量改进奖励（核心改进）===
        
    #     # 1. 完成率增量奖励（鼓励持续进步）
    #     current_completion_rate = len(self.completed_jobs) / total_jobs
    #     prev_completion_rate = getattr(self, 'prev_completion_rate', 0)
    #     completion_improvement = current_completion_rate - prev_completion_rate
        
    #     if completion_improvement > 0:
    #         # 使用指数奖励：小的持续改进 > 一次大的改进
    #         completion_reward = 8.0 * (np.exp(completion_improvement * 3) - 1)  # 指数增长
    #         reward += completion_reward
    #         debug_info['completion_improvement'] = completion_reward
        
    #     # 2. 延误改善奖励（相对改进）
    #     current_tardiness = sum(max(0, job.dispatched_time - job.due_date) for job in self.completed_jobs)
    #     prev_tardiness = getattr(self, 'prev_tardiness', 0)
        
    #     if prev_tardiness > 0 and current_tardiness < prev_tardiness:
    #         tardiness_improvement = (prev_tardiness - current_tardiness) / prev_tardiness  # 相对改进率
    #         tardiness_reward = 6.0 * np.tanh(tardiness_improvement * 2)  # 压缩到合理范围
    #         reward += tardiness_reward
    #         debug_info['tardiness_improvement'] = tardiness_reward
        
    #     # 3. 里程碑奖励（防止后期饱和）
    #     completion_milestone = len(self.completed_jobs)
    #     if completion_milestone >= getattr(self, 'prev_milestone', 0) + 5:  # 每完成5个作业
    #         milestone_bonus = 15.0
    #         reward += milestone_bonus
    #         self.prev_milestone = completion_milestone
    #         debug_info['milestone_bonus'] = milestone_bonus
        
    #     # 4. 交付满意度（保持但调整权重）
    #     delivery_satisfaction = self._calculate_delivery_satisfaction()
    #     delivery_reward = 1.5 * delivery_satisfaction  # 降低权重
    #     reward += delivery_reward
    #     debug_info['delivery_reward'] = delivery_reward
        
    #     # 5. 利用率奖励（目标导向）
    #     utilization = self.calculate_machine_utilization()
    #     target_util = 0.8  # 提高目标
    #     util_reward = 2.0 * np.exp(-((utilization - target_util) ** 2) / 0.1)  # 高斯奖励
    #     reward += util_reward
    #     debug_info['util_reward'] = util_reward
        
    #     # 6. 动作有效性奖励
    #     if any(['schedule' in action, 'dispatch' in action]):
    #         action_bonus = 0.3
    #         reward += action_bonus
    #         debug_info['action_bonus'] = action_bonus
        
    #     # 基础奖励
    #     reward += 1.0
    #     debug_info['base_reward'] = 1.0
        
    #     # 更新历史状态
    #     self.prev_completion_rate = current_completion_rate
    #     self.prev_tardiness = current_tardiness
        
    #     # 最终范围调整
    #     clipped_reward = float(np.clip(reward, 0, 50))  # 确保奖励为正
        
    #     debug_info['final_reward'] = clipped_reward
    #     return clipped_reward
   
    def _calculate_delivery_satisfaction(self):
        """计算配送要求满足度，返回0-1之间的值"""
        total_satisfaction = 0
        total_requirements = 0
        
        for distributor in self.distributors:
            if not distributor.delivery_requirements:
                continue
                
            requirement = distributor.delivery_requirements
            distributor_satisfaction = 0
            
            for due_time, ratio, weight in zip(requirement.due_times, 
                                            requirement.ratios, 
                                            requirement.weights):
                completed_jobs = [j for j in self.completed_jobs 
                                if j.dispatched_time <= due_time]
                completed_amount = sum(j.amount for j in completed_jobs)
                required_amount = ratio * distributor.total_amount
                
                if completed_amount >= required_amount:
                    satisfaction = 1.0  # 完全满足
                else:
                    satisfaction = completed_amount / required_amount  # 部分满足
                    
                distributor_satisfaction += satisfaction * weight
            
            total_satisfaction += distributor_satisfaction
            total_requirements += 1
        
        return total_satisfaction / total_requirements if total_requirements > 0 else 0
 

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
        """归一化所有机器的剩余时间方差"""
        loads = [m.remaining_time for m in self.machines]
        mean_load = np.mean(loads)
        if not loads or mean_load == 0:
            return 0.0
        return float(np.var(loads) / mean_load)



    def _update_job_states(self):
        """更新作业状态"""
        for job in self.available_jobs:
            if job.status == 'completed':
                if job not in self.completed_jobs:
                    self.completed_jobs.append(job)
                    self.completion_times[job.job_id] = self.t
                    job.completed_time = self.t
        #self.available_jobs = [j for j in self.available_jobs if j.status != 'dispatched']


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
            # 检查当前工序是否已经是最后一道工序
            if job.current_operation >= len(job.operations):
                # 作业已经完成所有工序，直接标记为完成
                job.status = 'completed'
                if job not in self.completed_jobs:
                    self.completed_jobs.append(job)
                    self.job_completed_this_step = True
            else:
                # 更新工序进度
                job.current_operation += 1
                self.operation_completed_this_step = True
                operation = job.operations[job.current_operation - 1] 
                operation.completed_time = self.t  # 设置工序完成时间
                
                # 检查是否所有工序都完成
                if job.current_operation >= len(job.operations):
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
                self.this_step_complete_jobs += 1
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
                if job.current_operation >= len(job.operations):
                    # 只在训练模式下显示警告信息
                    if hasattr(self.config, 'train_mode') and self.config.train_mode:
                        print(f"⚠️ 警告：作业{job.job_id}已无可调度工序，跳过调度。")
                    continue
                    
                current_op = job.operations[job.current_operation]
                if machine_id in current_op.available_machine_ids:
                        # 改变machine 状态
                        machine.status = 'busy'
                        machine.current_job = job.job_id
                        machine.remaining_time = current_op.processing_times[machine_id]
                        machine.total_busy_time += current_op.processing_times[machine_id]

                        # 改变job 状态
                        # 更新作业状态
                        job.status = 'processing'
                        self.machine_utilization= 1.0
                    


    def _process_dispatching(self, dispatch_action: Dict):
        """处理配送决策"""
        BASE_DELIVERY_TIME = 1  # 基础配送时间
        PER_JOB_TIME = 0        # 每多一个工件增加的配送时间
        for batch_id, job_ids in dispatch_action.items():
            # 只处理已完成加工的作
            flat_job_ids = list(chain.from_iterable(job_ids)) if job_ids and isinstance(job_ids[0], list) else job_ids

            dispatch_jobs = [j for j in self.completed_jobs if j.job_id in flat_job_ids and j.status == 'completed']
            
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
                    self.this_step_dispatch_jobs += 1
                    job.dispatched_time = self.t  # 配送完成时间
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
            'dispatched_jobs': self.dispatched_jobs,
            'machine_utilization': self.calculate_machine_utilization()
        }
