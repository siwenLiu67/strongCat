"""
PPO环境适配器 - 使PPO算法能够复用WarehouseEnvironment
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from environment import WarehouseEnvironment
from case_generator import FlexibleJobShopScenario
from config import Config


class PPOEnvironmentAdapter:
    """PPO环境适配器，将WarehouseEnvironment适配为PPO可用的接口"""
    
    def __init__(self, config: Config, scenario: FlexibleJobShopScenario):
        """初始化适配器
        
        Args:
            config: 配置对象
            scenario: 算例生成器实例
        """
        self.base_env = WarehouseEnvironment(config, scenario)
        self.scenario = scenario
        self.config = config
        
        # 状态和动作空间参数（与原始PPO保持一致）
        self.state_dim = 20
        self.action_dim = 1000
        
    def reset(self) -> np.ndarray:
        """重置环境并返回初始状态"""
        self.base_env.reset()
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """将WarehouseEnvironment的字典状态转换为PPO需要的numpy数组状态"""
        state_features = []
        
        # 获取基础环境状态
        base_state = self.base_env._get_state()
        
        # 时间特征
        state_features.append(self.base_env.t / 100.0)  # 归一化时间
        
        # 机器状态特征
        idle_machines = sum(1 for m in self.base_env.machines if m.status == "waiting")
        busy_machines = len(self.base_env.machines) - idle_machines
        state_features.extend([
            idle_machines / len(self.base_env.machines),
            busy_machines / len(self.base_env.machines)
        ])
        
        # 作业特征
        total_jobs = len(self.base_env.available_jobs) + len(self.base_env.completed_jobs)
        waiting_jobs = sum(1 for j in self.base_env.available_jobs if j.status == "waiting")
        processing_jobs = sum(1 for j in self.base_env.available_jobs if j.status == "processing")
        completed_jobs = len(self.base_env.completed_jobs)
        dispatched_jobs = len(self.base_env.dispatched_jobs)
        
        state_features.extend([
            waiting_jobs / max(total_jobs, 1),
            processing_jobs / max(total_jobs, 1),
            completed_jobs / max(total_jobs, 1),
            dispatched_jobs / max(total_jobs, 1)
        ])
        
        # 紧急度特征
        if self.base_env.available_jobs:
            urgencies = []
            for job in self.base_env.available_jobs:
                if job.status == "waiting":
                    # 计算剩余处理时间
                    remaining_operations = job.operations[job.current_operation:]
                    if remaining_operations:
                        remaining_time = sum(
                            min(op.processing_times.values()) if op.processing_times else 0
                            for op in remaining_operations
                        )
                    else:
                        remaining_time = 0
                    
                    due_date = getattr(job, 'due_date', 100)
                    urgency = max(0, (due_date - self.base_env.t - remaining_time)) / 100.0
                    urgencies.append(urgency)
            
            state_features.extend([
                np.mean(urgencies) if urgencies else 0,
                np.min(urgencies) if urgencies else 0,
                np.max(urgencies) if urgencies else 0
            ])
        else:
            state_features.extend([0, 0, 0])
        
        # 配送商负载特征
        for dist in self.base_env.distributors:
            dist_jobs = [j for j in self.base_env.available_jobs + self.base_env.completed_jobs 
                        if j.distributor_id == dist.distributor_id]
            completed_for_dist = [j for j in dist_jobs if j in self.base_env.completed_jobs]
            load_ratio = len(completed_for_dist) / max(len(dist_jobs), 1)
            state_features.append(load_ratio)
        
        # 填充到固定长度
        target_length = 20
        while len(state_features) < target_length:
            state_features.append(0.0)
        
        return np.array(state_features[:target_length], dtype=np.float32)
    
    def get_action_mask(self) -> np.ndarray:
        """获取有效动作掩码"""
        mask = np.zeros(self.action_dim, dtype=bool)
        
        # 获取索引化动作列表
        indexed_actions = self._get_indexed_actions()
        
        # 设置有效动作掩码
        for action_idx in range(len(indexed_actions)):
            if action_idx < self.action_dim:
                mask[action_idx] = True
        
        # 如果没有有效动作，允许等待
        if not np.any(mask):
            mask[-1] = True
        
        return mask
    
    def _get_indexed_actions(self) -> List[Tuple]:
        """获取索引化的动作列表"""
        actions = []
        action_idx = 0
        
        # 调度动作
        for job in self.base_env.available_jobs:
            if job.status == "waiting" and job.current_operation < len(job.operations):
                current_op = job.operations[job.current_operation]
                for machine_id in current_op.available_machine_ids:
                    if machine_id < len(self.base_env.machines):
                        machine = self.base_env.machines[machine_id]
                        if machine.status == "waiting" and action_idx < self.action_dim - 100:
                            actions.append(("schedule", job.job_id, machine_id, None))
                            action_idx += 1
        
        # 派遣动作
        completed_waiting = [j for j in self.base_env.completed_jobs 
                           if j not in self.base_env.dispatched_jobs]
        if completed_waiting:
            for dist in self.base_env.distributors:
                dist_jobs = [j for j in completed_waiting 
                           if j.distributor_id == dist.distributor_id]
                if dist_jobs and action_idx < self.action_dim - 10:
                    actions.append(("dispatch", None, None, dist.distributor_id))
                    action_idx += 1
        
        # 等待动作
        if action_idx < self.action_dim:
            actions.append(("wait", None, None, None))
        
        return actions
    
    def _decode_action(self, action_idx: int) -> Dict:
        """将PPO的整数动作解码为WarehouseEnvironment的字典动作"""
        indexed_actions = self._get_indexed_actions()
        
        if action_idx < len(indexed_actions):
            action_type, job_id, machine_id, distributor_id = indexed_actions[action_idx]
            
            if action_type == "schedule":
                return {'schedule': {job_id: machine_id}}
            elif action_type == "dispatch":
                # 找到该配送商的所有已完成未派遣作业
                completed_waiting = [j for j in self.base_env.completed_jobs 
                                   if j not in self.base_env.dispatched_jobs 
                                   and j.distributor_id == distributor_id]
                if completed_waiting:
                    job_ids = [job.job_id for job in completed_waiting]
                    return {'dispatch': {distributor_id: job_ids}}
                else:
                    return {'wait': {}}  # 如果没有作业可派遣，则等待
            else:  # wait
                return {'wait': {}}
        else:
            # 无效动作，等待
            return {'wait': {}}
    
    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, dict]:
        """执行PPO动作并返回结果
        
        Args:
            action_idx: PPO的整数动作索引
            
        Returns:
            Tuple[np.ndarray, float, bool, dict]: 下一个状态，奖励，是否完成，信息
        """
        # 解码动作
        env_action = self._decode_action(action_idx)
        
        # 执行基础环境动作
        next_state_dict, reward, done, info = self.base_env.step(env_action)
        
        # 转换状态格式
        next_state_np = self._get_state()
        
        return next_state_np, reward, done, info
    
    def _is_done(self) -> bool:
        """检查是否完成"""
        return self.base_env.done
    
    @property
    def jobs(self):
        """获取作业列表"""
        return self.base_env.available_jobs + self.base_env.completed_jobs
    
    @property
    def machines(self):
        """获取机器列表"""
        return self.base_env.machines
    
    @property
    def distributors(self):
        """获取配送商列表"""
        return self.base_env.distributors
    
    @property
    def completed_jobs(self):
        """获取已完成作业列表"""
        return self.base_env.completed_jobs
    
    @property
    def dispatched_jobs(self):
        """获取已派遣作业列表"""
        return self.base_env.dispatched_jobs
    
    @property
    def current_time(self):
        """获取当前时间"""
        return self.base_env.t


# 兼容性包装器，保持与原始PPO环境相同的接口
class FJSSPEnvironment:
    """兼容性包装器，保持与原始PPO环境相同的接口"""
    
    def __init__(self, scenario: FlexibleJobShopScenario):
        """初始化兼容性环境
        
        Args:
            scenario: 算例生成器实例
        """
        config = Config()
        self.adapter = PPOEnvironmentAdapter(config, scenario)
        self.scenario = scenario
        self.action_dim = self.adapter.action_dim
        
    def reset(self):
        """重置环境"""
        return self.adapter.reset()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """执行动作"""
        return self.adapter.step(action)
    
    def get_action_mask(self) -> np.ndarray:
        """获取有效动作掩码"""
        return self.adapter.get_action_mask()
    
    def _is_done(self) -> bool:
        """检查是否完成"""
        return self.adapter._is_done()
    
    @property
    def jobs(self):
        """获取作业列表"""
        return self.adapter.jobs
    
    @property
    def machines(self):
        """获取机器列表"""
        return self.adapter.machines
    
    @property
    def distributors(self):
        """获取配送商列表"""
        return self.adapter.distributors
    
    @property
    def completed_jobs(self):
        """获取已完成作业列表"""
        return self.adapter.completed_jobs
    
    @property
    def dispatched_jobs(self):
        """获取已派遣作业列表"""
        return self.adapter.dispatched_jobs
    
    @property
    def current_time(self):
        """获取当前时间"""
        return self.adapter.current_time
