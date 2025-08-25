"""
HIRO环境适配器 - 使HIRO算法能够复用WarehouseEnvironment
参考DQNEnvironmentAdapter的实现模式
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from environment import WarehouseEnvironment
from case_generator import FlexibleJobShopScenario
from config import Config


class HIROEnvironmentAdapter:
    """HIRO环境适配器，将WarehouseEnvironment适配为HIRO可用的接口"""
    
    def __init__(self, config: Config, scenario: FlexibleJobShopScenario):
        """初始化适配器
        
        Args:
            config: 配置对象
            scenario: 算例生成器实例
        """
        self.base_env = WarehouseEnvironment(config, scenario)
        self.scenario = scenario
        self.config = config
        
        # HIRO特定的状态和动作空间参数
        self.state_dim = self._calculate_state_dim()
        self.action_dim = self._calculate_action_dim()
        self.goal_dim = 32  # 子目标维度，与原始HIRO保持一致
    
    def _calculate_state_dim(self) -> int:
        """计算状态空间维度"""
        # 作业状态 + 机器状态 + 时间信息 + 派遣状态
        job_features = len(self.scenario.jobs) * 4  # 进度、剩余时间、延误、完成状态
        machine_features = len(self.scenario.machines) * 3  # 忙碌状态、负载、可用时间
        time_features = 3  # 当前时间、平均延误、完工进度
        dispatch_features = len(self.scenario.distributors) * 2  # 批次数量、等待作业数
        
        return job_features + machine_features + time_features + dispatch_features
    
    def _calculate_action_dim(self) -> int:
        """计算动作空间维度"""
        # 调度动作 + 派遣动作（连续动作空间）
        schedule_actions = 3  # 作业选择概率、机器选择概率、优先级权重
        dispatch_actions = 2  # 批次组建决策、派遣时机决策
        
        return schedule_actions + dispatch_actions
    
    def reset(self) -> np.ndarray:
        """重置环境并返回初始状态"""
        self.base_env.reset()
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """获取当前状态（转换为numpy数组格式）"""
        state = []
        
        # 获取基础环境状态
        base_state = self.base_env._get_state()
        
        # 作业状态特征
        for job in self.base_env.available_jobs:
            if job.job_id in [j.job_id for j in self.base_env.dispatched_jobs]:
                state.extend([1.0, 0.0, 0.0, 1.0])  # 已配送
            elif job.job_id in [j.job_id for j in self.base_env.completed_jobs]:
                state.extend([1.0, 0.0, 0.0, 0.0])  # 已完成
            else:
                # 计算进度
                progress = job.current_operation / len(job.operations) if job.operations else 0
                
                # 剩余处理时间估计
                remaining_ops = len(job.operations) - job.current_operation
                avg_proc_time = np.mean([min(op.processing_times.values()) 
                                       for op in job.operations]) if job.operations else 0
                remaining_time = remaining_ops * avg_proc_time
                
                # 延误计算
                due_date = getattr(job, 'due_date', 1000)
                tardiness = max(0, self.current_time - due_date) / 100.0  # 归一化
                
                status_code = 1.0 if job.status == 'processing' else 0.0
                state.extend([progress, remaining_time / 100.0, tardiness, status_code])
        
                # 机器状态特征
        for machine in self.base_env.machines:
            busy = 1.0 if machine.status == 'busy' else 0.0
            
            # 计算机器负载 - 正在处理的作业数量
            processing_jobs = [j for j in self.base_env.available_jobs 
                             if j.status == 'processing']
            load = len([j for j in processing_jobs 
                       if any(op.operation_id == machine.current_job 
                             for op in j.operations)]) / max(len(self.scenario.jobs), 1)
            
            available_time = machine.remaining_time / 100.0 if hasattr(machine, 'remaining_time') else 0.0
            
            state.extend([busy, load, available_time])
        
        # 时间特征
        time_progress = self.current_time / 200.0  # 归一化时间
        avg_tardiness = np.mean([max(0, self.current_time - getattr(job, 'due_date', 1000)) 
                                for job in self.base_env.available_jobs]) / 100.0 if self.base_env.available_jobs else 0
        completion_rate = len(self.base_env.completed_jobs) / len(self.base_env.available_jobs) if self.base_env.available_jobs else 0
        
        state.extend([time_progress, avg_tardiness, completion_rate])
        
        # 派遣状态特征
        for distributor in self.base_env.distributors:
            dist_jobs = [job for job in self.base_env.available_jobs 
                        if getattr(job, 'distributor_id', 0) == distributor.distributor_id]
            batch_count = len([b for b in getattr(self.base_env, 'batches', []) 
                              if b.get('distributor') == distributor.distributor_id])
            waiting_jobs = len([job for job in dist_jobs if job.job_id in [j.job_id for j in self.base_env.completed_jobs]])
            
            state.extend([batch_count / 10.0, waiting_jobs / max(len(dist_jobs), 1)])
        
        return np.array(state, dtype=np.float32)
    
    def get_achieved_goal(self) -> np.ndarray:
        """获取当前达成的目标"""
        # 基于当前状态计算达成的目标
        completion_rate = len(self.base_env.completed_jobs) / len(self.base_env.available_jobs) if self.base_env.available_jobs else 0
        avg_machine_util = self.base_env.calculate_machine_utilization()
        
        avg_tardiness = np.mean([max(0, self.current_time - getattr(job, 'due_date', 1000)) 
                               for job in self.base_env.available_jobs]) / 100.0 if self.base_env.available_jobs else 0
        
        dispatch_efficiency = len(getattr(self.base_env, 'batches', [])) / max(len(self.base_env.completed_jobs), 1) if self.base_env.completed_jobs else 0
        
        # 组合成目标向量
        achieved_goal = np.array([
            completion_rate,
            avg_machine_util,
            -avg_tardiness,  # 负值因为我们希望延误更小
            dispatch_efficiency
        ] + [0] * (self.goal_dim - 4), dtype=np.float32)  # 填充到目标维度
        
        return achieved_goal
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行HIRO动作并返回结果
        
        Args:
            action: HIRO的连续动作向量
            
        Returns:
            Tuple[np.ndarray, float, bool, dict]: 下一个状态，奖励，是否完成，信息
        """
        # 将连续动作转换为环境动作
        env_action = self._decode_action(action)
        
        # 执行基础环境动作
        next_state_dict, reward, done, info = self.base_env.step(env_action)
        
        # 转换状态格式
        next_state_np = self.get_state()
        
        return next_state_np, reward, done, info
    
    def _decode_action(self, action: np.ndarray) -> Dict:
        """将HIRO的连续动作解码为WarehouseEnvironment的字典动作"""
        action = np.clip(action, -1, 1)  # 确保动作在有效范围内
        
        # 解析动作
        schedule_action = action[:3]
        dispatch_action = action[3:5]
        
        # 构建调度动作
        schedule_dict = {}
        if len(self.base_env.available_jobs) > 0:
            # 简化的调度逻辑 - 在实际实现中需要更复杂的决策
            waiting_jobs = [j for j in self.base_env.available_jobs if j.status == 'waiting']
            if waiting_jobs:
                # 选择作业和机器（简化实现）
                job_idx = int((schedule_action[0] + 1) / 2 * (len(waiting_jobs) - 1))
                selected_job = waiting_jobs[job_idx]
                
                if selected_job.current_operation < len(selected_job.operations):
                    current_op = selected_job.operations[selected_job.current_operation]
                    if current_op.available_machine_ids:
                        machine_idx = int((schedule_action[1] + 1) / 2 * (len(current_op.available_machine_ids) - 1))
                        selected_machine = current_op.available_machine_ids[machine_idx]
                        
                        schedule_dict[selected_job.job_id] = selected_machine
        
        # 构建派遣动作
        dispatch_dict = {}
        if len(self.base_env.completed_jobs) > 0:
            # 简化的派遣逻辑
            batch_threshold = (dispatch_action[0] + 1) / 2
            dispatch_urgency = (dispatch_action[1] + 1) / 2
            
            if dispatch_urgency > 0.7 or len(self.base_env.completed_jobs) >= 3:
                # 按配送商分组派遣
                for distributor in self.base_env.distributors:
                    dist_jobs = [j for j in self.base_env.completed_jobs 
                               if j.distributor_id == distributor.distributor_id and j not in self.base_env.dispatched_jobs]
                    if dist_jobs and (len(dist_jobs) >= 2 or batch_threshold > 0.8):
                        dispatch_dict[distributor.distributor_id] = [j.job_id for j in dist_jobs]
        
        if schedule_dict:
            return {'schedule': schedule_dict}
        elif dispatch_dict:
            return {'dispatch': dispatch_dict}
        else:
            return {'wait': {}}
    
    # 属性代理 - 暴露基础环境的所有属性
    @property
    def current_time(self):
        """获取当前时间"""
        return self.base_env.t
    
    @property
    def total_weighted_tardiness(self):
        """获取总加权延迟时间"""
        return self.base_env.total_weighted_tardiness
    
    @property
    def tardy_penalty(self):
        """获取延迟惩罚"""
        return self.base_env.tardy_penalty
    
    @property
    def dynamic_jobs_arrived(self):
        """获取已到达的动态作业数"""
        return self.base_env.dynamic_jobs_arrived
    
    @property
    def remaining_dynamic_jobs(self):
        """获取剩余动态作业数"""
        return self.base_env.remaining_dynamic_jobs
    
    @property
    def arrival_events(self):
        """获取到达事件记录"""
        return self.base_env.arrival_events
    
    @property
    def completion_times(self):
        """获取完成时间记录"""
        return self.base_env.completion_times
    
    @property
    def machine_utilization(self):
        """获取机器利用率统计"""
        return self.base_env.machine_utilization
    
    @property
    def initial_jobs(self):
        """获取初始作业列表"""
        return self.base_env.initial_jobs
    
    @property
    def available_jobs(self):
        """获取可用作业列表"""
        return self.base_env.available_jobs
    
    @property
    def completed_jobs(self):
        """获取已完成作业列表"""
        return self.base_env.completed_jobs
    
    @property
    def dispatched_jobs(self):
        """获取已派遣作业列表"""
        return self.base_env.dispatched_jobs
    
    @property
    def machines(self):
        """获取机器列表"""
        return self.base_env.machines
    
    @property
    def distributors(self):
        """获取配送商列表"""
        return self.base_env.distributors
    
    @property
    def done(self):
        """获取是否完成"""
        return self.base_env.done
    
    def get_dynamic_arrival_statistics(self):
        """获取动态到达统计信息"""
        return self.base_env.get_dynamic_arrival_statistics()
    
    def calculate_machine_utilization(self):
        """计算机器利用率"""
        return self.base_env.calculate_machine_utilization()
    
    def calculate_operation_progress_ratio(self):
        """计算作业进度比例"""
        return self.base_env.calculate_operation_progress_ratio()
    
    def calculate_machine_load_variance(self):
        """计算机器负载方差"""
        return self.base_env.calculate_machine_load_variance()


# 兼容性包装器，保持与原始HIRO环境相同的接口
class ScheduleEnvironment:
    """兼容性包装器，保持与原始HIRO环境相同的接口"""
    
    def __init__(self, scenario: FlexibleJobShopScenario):
        """初始化兼容性环境
        
        Args:
            scenario: 算例生成器实例
        """
        config = Config()
        self.adapter = HIROEnvironmentAdapter(config, scenario)
        self.scenario = scenario
        
        # 保持原始接口
        self.state_dim = self.adapter.state_dim
        self.action_dim = self.adapter.action_dim
        self.goal_dim = self.adapter.goal_dim
        
    def reset(self):
        """重置环境"""
        return self.adapter.reset()
    
    def get_state(self):
        """获取状态"""
        return self.adapter.get_state()
    
    def get_achieved_goal(self):
        """获取达成目标"""
        return self.adapter.get_achieved_goal()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作"""
        return self.adapter.step(action)
    
    # 属性代理
    @property
    def jobs(self):
        return self.adapter.base_env.available_jobs + self.adapter.base_env.completed_jobs
    
    @property
    def machines(self):
        return self.adapter.machines
    
    @property
    def distributors(self):
        return self.adapter.distributors
    
    @property
    def current_time(self):
        return self.adapter.current_time
    
    @property
    def total_weighted_tardiness(self):
        return self.adapter.total_weighted_tardiness
    
    @property
    def tardy_penalty(self):
        return self.adapter.tardy_penalty
