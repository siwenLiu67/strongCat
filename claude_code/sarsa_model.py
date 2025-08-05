"""
使用SARSA算法解决集成柔性作业车间调度与派遣问题(IFJSSP-DP)
SARSA: State-Action-Reward-State-Action 时序差分学习算法
"""

import numpy as np
import random
import pickle
import time
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import math

from case_generator import FlexibleJobShopScenario
from config import Config
from data_structures import Job, Operation, Machine


class StateEncoder:
    """状态编码器 - 将连续状态离散化"""
    
    def __init__(self, num_jobs: int, num_machines: int, num_distributors: int):
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.num_distributors = num_distributors
        
        # 减少状态空间复杂度
        self.time_bins = 5
        self.machine_state_bins = 3
        self.job_state_bins = 3
        self.urgency_bins = 3
        
        # 计算状态空间大小
        self.state_space_size = (
            self.time_bins * 
            (self.machine_state_bins ** 2) * 
            (self.job_state_bins ** 4) * 
            (self.urgency_bins ** 3)
        )
        
        print(f"状态空间大小: {self.state_space_size}")
    
    def encode_state(self, raw_state: np.ndarray) -> int:
        """将原始状态编码为离散状态ID"""
        # 确保状态向量长度正确
        if len(raw_state) < 10:
            raw_state = np.pad(raw_state, (0, 10 - len(raw_state)), 'constant')
        
        # 离散化各个特征，添加边界保护
        time_idx = max(0, min(int(raw_state[0] * self.time_bins), self.time_bins - 1))
        
        machine_idle_idx = max(0, min(int(raw_state[1] * self.machine_state_bins), self.machine_state_bins - 1))
        machine_busy_idx = max(0, min(int(raw_state[2] * self.machine_state_bins), self.machine_state_bins - 1))
        
        job_waiting_idx = max(0, min(int(raw_state[3] * self.job_state_bins), self.job_state_bins - 1))
        job_processing_idx = max(0, min(int(raw_state[4] * self.job_state_bins), self.job_state_bins - 1))
        job_completed_idx = max(0, min(int(raw_state[5] * self.job_state_bins), self.job_state_bins - 1))
        job_dispatched_idx = max(0, min(int(raw_state[6] * self.job_state_bins), self.job_state_bins - 1))
        
        urgency_mean_idx = max(0, min(int(raw_state[7] * self.urgency_bins), self.urgency_bins - 1))
        urgency_min_idx = max(0, min(int(raw_state[8] * self.urgency_bins), self.urgency_bins - 1))
        urgency_max_idx = max(0, min(int(raw_state[9] * self.urgency_bins), self.urgency_bins - 1))
        
        # 计算复合状态ID
        state_id = (
            time_idx * (self.machine_state_bins ** 2) * (self.job_state_bins ** 4) * (self.urgency_bins ** 3) +
            machine_idle_idx * self.machine_state_bins * (self.job_state_bins ** 4) * (self.urgency_bins ** 3) +
            machine_busy_idx * (self.job_state_bins ** 4) * (self.urgency_bins ** 3) +
            job_waiting_idx * (self.job_state_bins ** 3) * (self.urgency_bins ** 3) +
            job_processing_idx * (self.job_state_bins ** 2) * (self.urgency_bins ** 3) +
            job_completed_idx * self.job_state_bins * (self.urgency_bins ** 3) +
            job_dispatched_idx * (self.urgency_bins ** 3) +
            urgency_mean_idx * (self.urgency_bins ** 2) +
            urgency_min_idx * self.urgency_bins +
            urgency_max_idx
        )
        
        return max(0, min(state_id, self.state_space_size - 1))


class ActionEncoder:
    """动作编码器"""
    
    def __init__(self, num_jobs: int, num_machines: int, num_distributors: int):
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.num_distributors = num_distributors
        
        # 动作类型
        self.schedule_actions = num_jobs * num_machines
        self.dispatch_actions = num_distributors
        self.wait_actions = 1
        
        self.action_space_size = self.schedule_actions + self.dispatch_actions + self.wait_actions
        print(f"动作空间大小: {self.action_space_size}")
    
    def encode_action(self, action_type: str, job_id: Optional[int] = None, 
                     machine_id: Optional[int] = None, distributor_id: Optional[int] = None) -> int:
        """编码动作"""
        if action_type == "schedule":
            if job_id is None or machine_id is None:
                return self.action_space_size - 1  # 返回等待动作
            return job_id * self.num_machines + machine_id
        elif action_type == "dispatch":
            if distributor_id is None:
                return self.action_space_size - 1  # 返回等待动作
            return self.schedule_actions + distributor_id
        else:  # wait
            return self.schedule_actions + self.dispatch_actions
    
    def decode_action(self, action_id: int) -> Tuple[str, Optional[int], Optional[int], Optional[int]]:
        """解码动作"""
        if action_id < 0 or action_id >= self.action_space_size:
            return "wait", None, None, None
            
        if action_id < self.schedule_actions:
            job_id = action_id // self.num_machines
            machine_id = action_id % self.num_machines
            return "schedule", job_id, machine_id, None
        elif action_id < self.schedule_actions + self.dispatch_actions:
            distributor_id = action_id - self.schedule_actions
            return "dispatch", None, None, distributor_id
        else:
            return "wait", None, None, None


class FJSSPEnvironment:
    """FJSSP-DP环境（适配SARSA算法）"""
    
    def __init__(self, scenario: FlexibleJobShopScenario):
        self.scenario = scenario
        self.jobs = scenario.jobs
        self.machines = scenario.machines
        self.distributors = scenario.distributors
        
        # 状态和动作编码器
        self.state_encoder = StateEncoder(len(self.jobs), len(self.machines), len(self.distributors))
        self.action_encoder = ActionEncoder(len(self.jobs), len(self.machines), len(self.distributors))
        
        # 内部跟踪变量
        self.job_start_times = {}  # 跟踪作业开始时间
        self.job_arrival_flags = {}  # 跟踪作业是否已到达
        
        self.reset()
    
    def reset(self):
        """重置环境"""
        self.current_time = 0
        self.completed_jobs = []
        self.dispatched_jobs = []
        self.pending_jobs = self.jobs.copy()
        self.available_jobs = []
        
        # 重置内部跟踪变量
        self.job_start_times = {}
        self.job_arrival_flags = {job.job_id: False for job in self.jobs}
        
        # 重置机器状态
        for machine in self.machines:
            machine.status = "waiting"
            machine.current_job = -1
            machine.remaining_time = 0.0
            # 确保机器有必要的属性
            if not hasattr(machine, 'processed_jobs'):
                machine.processed_jobs = []
            if not hasattr(machine, 'total_busy_time'):
                machine.total_busy_time = 0.0
            if not hasattr(machine, 'total_idle_time'):
                machine.total_idle_time = 0.0
        
        # 重置作业状态
        for job in self.jobs:
            job.status = "waiting"
            job.current_operation = 0
            job.completed_time = 0.0
            job.dispatched_time = 0.0
            # 不再设置arrival_time和start_time属性
        
        self._update_available_jobs()
        return self.get_state()
    
    def get_state(self) -> int:
        """获取当前状态（离散化）"""
        raw_state = self._get_raw_state()
        return self.state_encoder.encode_state(raw_state)
    
    def _get_raw_state(self) -> np.ndarray:
        """获取原始连续状态"""
        state_features = []
        
        # 时间特征
        state_features.append(min(self.current_time / 100.0, 1.0))
        
        # 机器状态特征
        idle_machines = sum(1 for m in self.machines if getattr(m, 'status', 'waiting') == "waiting")
        busy_machines = len(self.machines) - idle_machines
        state_features.extend([
            idle_machines / max(len(self.machines), 1),
            busy_machines / max(len(self.machines), 1)
        ])
        
        # 作业特征
        total_jobs = len(self.jobs)
        waiting_jobs = sum(1 for j in self.available_jobs if getattr(j, 'status', 'waiting') == "waiting")
        processing_jobs = sum(1 for j in self.jobs if getattr(j, 'status', 'waiting') == "processing")
        completed_jobs = len(self.completed_jobs)
        dispatched_jobs = len(self.dispatched_jobs)
        
        state_features.extend([
            waiting_jobs / max(total_jobs, 1),
            processing_jobs / max(total_jobs, 1),
            completed_jobs / max(total_jobs, 1),
            dispatched_jobs / max(total_jobs, 1)
        ])
        
        # 紧急度特征
        if self.available_jobs:
            urgencies = []
            for job in self.available_jobs:
                try:
                    current_op_idx = getattr(job, 'current_operation', 0)
                    if current_op_idx < len(job.operations):
                        remaining_operations = job.operations[current_op_idx:]
                        remaining_time = sum(
                            min(op.processing_times.values()) if op.processing_times else 0
                            for op in remaining_operations
                        )
                    else:
                        remaining_time = 0
                    
                    due_date = getattr(job, 'due_date', 100)
                    urgency = max(0, min(1.0, (due_date - self.current_time - remaining_time) / max(due_date, 1)))
                    urgencies.append(urgency)
                except:
                    urgencies.append(0.5)  # 默认中等紧急度
            
            if urgencies:
                state_features.extend([
                    np.mean(urgencies),
                    np.min(urgencies),
                    np.max(urgencies)
                ])
            else:
                state_features.extend([0.0, 0.0, 0.0])
        else:
            state_features.extend([0.0, 0.0, 0.0])
        
        return np.array(state_features, dtype=np.float32)
    
    def get_valid_actions(self) -> List[int]:
        """获取当前有效的动作ID列表"""
        valid_actions = []
        
        try:
            # 调度动作
            for job in self.available_jobs:
                if (hasattr(job, 'status') and job.status == "waiting" and 
                    hasattr(job, 'current_operation') and job.current_operation < len(job.operations)):
                    
                    current_op = job.operations[job.current_operation]
                    for machine_id in current_op.available_machine_ids:
                        if machine_id < len(self.machines):
                            machine = self.machines[machine_id]
                            if getattr(machine, 'status', 'waiting') == "waiting":
                                action_id = self.action_encoder.encode_action("schedule", job.job_id, machine_id)
                                valid_actions.append(action_id)
            
            # 派遣动作
            completed_waiting = [j for j in self.completed_jobs if j not in self.dispatched_jobs]
            if completed_waiting:
                for dist in self.distributors:
                    dist_jobs = [j for j in completed_waiting if hasattr(j, 'distributor_id') and j.distributor_id == dist.distributor_id]
                    if dist_jobs:
                        action_id = self.action_encoder.encode_action("dispatch", distributor_id=dist.distributor_id)
                        valid_actions.append(action_id)
            
            # 等待动作
            wait_action_id = self.action_encoder.encode_action("wait")
            valid_actions.append(wait_action_id)
        
        except Exception as e:
            print(f"获取有效动作时出错: {e}")
            # 如果出错，至少返回等待动作
            wait_action_id = self.action_encoder.encode_action("wait")
            valid_actions = [wait_action_id]
        
        return valid_actions if valid_actions else [self.action_encoder.action_space_size - 1]
    
    def step(self, action_id: int) -> Tuple[int, float, bool, dict]:
        """执行动作"""
        reward = 0
        info = {}
        
        try:
            # 解码动作
            action_type, job_id, machine_id, distributor_id = self.action_encoder.decode_action(action_id)
            
            # 执行动作
            if action_type == "schedule":
                if job_id is not None and machine_id is not None:
                    reward += self._schedule_job(job_id, machine_id)
                else:
                    reward -= 1  # 或其他惩罚值，表示无效动作
            elif action_type == "dispatch":
                if distributor_id is not None:
                    reward += self._dispatch_jobs(distributor_id)
                else:
                    reward -= 1  # 或其他惩罚值，表示无效动作
            else:  # wait
                reward -= 0.1
            
            # 推进时间
            self._advance_time()
            self._update_available_jobs()
            
            # 计算延误惩罚
            tardiness_penalty = self._calculate_tardiness_penalty()
            reward -= tardiness_penalty
            
            # 检查完成条件
            done = self._is_done()
            
            next_state = self.get_state()
            
        except Exception as e:
            print(f"执行动作时出错: {e}")
            reward = -1
            done = True
            next_state = self.get_state()
        
        return next_state, reward, done, info
    
    def _update_available_jobs(self):
        """更新可用作业列表"""
        try:
            # 清空当前可用作业列表
            self.available_jobs.clear()
            
            # 简化逻辑：所有作业在时间0就可用
            for job in self.pending_jobs[:]:
                if not self.job_arrival_flags[job.job_id]:
                    self.job_arrival_flags[job.job_id] = True
                    self.available_jobs.append(job)
                    self.pending_jobs.remove(job)
            
            # 添加当前等待中的作业（完成上一工序后等待下一工序）
            for job in self.jobs:
                if (getattr(job, 'status', 'waiting') == "waiting" and 
                    job not in self.pending_jobs and 
                    job not in self.available_jobs and
                    job not in self.completed_jobs):
                    self.available_jobs.append(job)
        except Exception as e:
            print(f"更新可用作业时出错: {e}")
    
    def _schedule_job(self, job_id: int, machine_id: int) -> float:
        """调度作业到机器"""
        try:
            # 验证作业和机器
            job = next((j for j in self.available_jobs if hasattr(j, 'job_id') and j.job_id == job_id), None)
            if not job or getattr(job, 'status', 'waiting') != "waiting":
                return -1
            
            if machine_id is None or machine_id >= len(self.machines):
                return -1
            
            machine = self.machines[machine_id]
            if getattr(machine, 'status', 'waiting') != "waiting":
                return -1
            
            current_op_idx = getattr(job, 'current_operation', 0)
            if current_op_idx >= len(job.operations):
                return -1
            
            current_op = job.operations[current_op_idx]
            if machine_id not in current_op.available_machine_ids:
                return -1
            
            processing_time = current_op.processing_times.get(machine_id, 0)
            if processing_time <= 0:
                return -1
            
            # 执行调度
            machine.current_job = job_id
            machine.remaining_time = processing_time
            machine.status = "busy"
            
            job.status = "processing"
            # 使用内部跟踪而不是job的属性
            if job_id not in self.job_start_times:
                self.job_start_times[job_id] = self.current_time
            
            # 从可用作业列表中移除
            if job in self.available_jobs:
                self.available_jobs.remove(job)
            
            # 计算奖励
            due_date = getattr(job, 'due_date', 100)
            urgency = max(0, (due_date - self.current_time)) / max(due_date, 1)
            return 2.0 + urgency
            
        except Exception as e:
            print(f"调度作业时出错: {e}")
            return -1
    
    def _dispatch_jobs(self, distributor_id: int) -> float:
        """派遣作业"""
        try:
            completed_waiting = [j for j in self.completed_jobs 
                               if j not in self.dispatched_jobs 
                               and hasattr(j, 'distributor_id') and j.distributor_id == distributor_id]
            
            if not completed_waiting:
                return -0.5
            
            reward = 0
            for job in completed_waiting:
                job.dispatched_time = self.current_time
                job.status = "dispatched"
                self.dispatched_jobs.append(job)
                
                due_date = getattr(job, 'due_date', 100)
                completed_time = getattr(job, 'completed_time', self.current_time)
                if completed_time <= due_date:
                    reward += 1.0
                else:
                    reward += 0.5
            
            reward += len(completed_waiting) * 0.2
            return reward
            
        except Exception as e:
            print(f"派遣作业时出错: {e}")
            return -0.5
    
    def _advance_time(self):
        """推进时间"""
        try:
            self.current_time += 1
            
            for machine in self.machines:
                if getattr(machine, 'status', 'waiting') == "busy":
                    current_remaining = getattr(machine, 'remaining_time', 0)
                    machine.remaining_time = max(0, current_remaining - 1)
                    
                    if machine.remaining_time <= 0:
                        job_id = getattr(machine, 'current_job', -1)
                        job = next((j for j in self.jobs if hasattr(j, 'job_id') and j.job_id == job_id), None)
                        
                        if job:
                            current_op = getattr(job, 'current_operation', 0)
                            job.current_operation = current_op + 1
                            
                            if job.current_operation >= len(job.operations):
                                # 作业完成
                                job.status = "completed"
                                job.completed_time = self.current_time
                                if job not in self.completed_jobs:
                                    self.completed_jobs.append(job)
                            else:
                                # 还有后续工序，返回等待状态
                                job.status = "waiting"
                        
                        # 重置机器状态
                        machine.status = "waiting"
                        machine.current_job = -1
                        machine.remaining_time = 0.0
                        
                        # 记录处理过的作业
                        if hasattr(machine, 'processed_jobs'):
                            machine.processed_jobs.append(job_id)
                            
        except Exception as e:
            print(f"推进时间时出错: {e}")
    
    def _calculate_tardiness_penalty(self) -> float:
        """计算延误惩罚"""
        try:
            penalty = 0
            for job in self.completed_jobs:
                due_date = getattr(job, 'due_date', 100)
                completed_time = getattr(job, 'completed_time', 0)
                if completed_time > due_date:
                    penalty += (completed_time - due_date) * 0.1
            return penalty
        except Exception as e:
            print(f"计算延误惩罚时出错: {e}")
            return 0
    
    def _is_done(self) -> bool:
        """检查是否完成"""
        try:
            all_jobs_completed = len(self.completed_jobs) == len(self.jobs)
            all_jobs_dispatched = len(self.dispatched_jobs) == len(self.jobs)
            timeout = self.current_time > 150  # 减少超时时间
            
            return (all_jobs_completed and all_jobs_dispatched) or timeout
        except Exception as e:
            print(f"检查完成条件时出错: {e}")
            return True


class SARSAAgent:
    """SARSA智能体"""
    
    def __init__(self, state_space_size: int, action_space_size: int, config: Config):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.config = config
        
        # SARSA超参数
        self.alpha = 0.2  # 提高学习率
        self.gamma = 0.9  # 降低折扣因子
        self.epsilon_start = 0.8
        self.epsilon_end = 0.05
        self.epsilon_decay = 0.995
        self.epsilon = self.epsilon_start
        
        # Q值表（使用字典存储稀疏表）
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # 统计信息
        self.episodes_trained = 0
        self.total_updates = 0
        
        print(f"SARSA Agent初始化:")
        print(f"  状态空间: {state_space_size}")
        print(f"  动作空间: {action_space_size}")
        print(f"  学习率: {self.alpha}")
        print(f"  折扣因子: {self.gamma}")
    
    def select_action(self, state: int, valid_actions: List[int]) -> int:
        """ε-贪婪策略选择动作"""
        if not valid_actions:
            return self.action_space_size - 1
            
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            if state not in self.q_table:
                return random.choice(valid_actions)
            
            best_value = float('-inf')
            best_actions = []
            
            for action in valid_actions:
                q_value = self.q_table[state][action]
                if q_value > best_value:
                    best_value = q_value
                    best_actions = [action]
                elif abs(q_value - best_value) < 1e-10:  # 处理浮点数精度问题
                    best_actions.append(action)
            
            return random.choice(best_actions)
    
    def update_q_value(self, state: int, action: int, reward: float, 
                       next_state: int, next_action: int):
        """SARSA更新规则"""
        current_q = self.q_table[state][action]
        next_q = self.q_table[next_state][next_action]
        
        # SARSA更新公式: Q(s,a) = Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        td_error = reward + self.gamma * next_q - current_q
        self.q_table[state][action] = current_q + self.alpha * td_error
        
        self.total_updates += 1
    
    def update_epsilon(self):
        """更新探索率"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def finish_episode(self):
        """完成一个episode"""
        self.episodes_trained += 1
        self.update_epsilon()
    
    def save_model(self, filepath: str):
        """保存Q表"""
        try:
            q_table_dict = {}
            for state, actions in self.q_table.items():
                q_table_dict[state] = dict(actions)
                
            model_data = {
                'q_table': q_table_dict,
                'epsilon': self.epsilon,
                'episodes_trained': self.episodes_trained,
                'hyperparameters': {
                    'alpha': self.alpha,
                    'gamma': self.gamma,
                    'epsilon_start': self.epsilon_start,
                    'epsilon_end': self.epsilon_end,
                    'epsilon_decay': self.epsilon_decay
                }
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
        except Exception as e:
            print(f"保存模型时出错: {e}")
    
    def load_model(self, filepath: str):
        """加载Q表"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            loaded_q_table = model_data['q_table']
            self.q_table = defaultdict(lambda: defaultdict(float))
            for state, actions in loaded_q_table.items():
                for action, q_value in actions.items():
                    self.q_table[state][action] = q_value
            
            self.epsilon = model_data['epsilon']
            self.episodes_trained = model_data['episodes_trained']
        except Exception as e:
            print(f"加载模型时出错: {e}")


def main():
    """主训练函数"""
    # 配置参数
    config = Config()
    config.num_jobs = 3  # 进一步减少作业数量
    config.num_machines = 2
    config.num_distributors = 2
    config.min_operations = 2
    config.max_operations = 2
    
    print("生成FJSP-DP场景...")
    scenario = FlexibleJobShopScenario(config=config)
    
    print(f"场景信息:")
    print(f"- 作业数量: {len(scenario.jobs)}")
    print(f"- 机器数量: {len(scenario.machines)}")
    print(f"- 配送商数量: {len(scenario.distributors)}")
    
    # 创建环境和智能体
    env = FJSSPEnvironment(scenario)
    agent = SARSAAgent(
        env.state_encoder.state_space_size,
        env.action_encoder.action_space_size,
        config
    )
    
    # 训练参数
    episodes = 200
    stats = defaultdict(list)
    
    print(f"\n开始SARSA训练 {episodes} episodes...")
    start_time = time.time()
    
    for episode in range(episodes):
        try:
            # 重置环境
            state = env.reset()
            
            # 选择初始动作
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions)
            
            episode_reward = 0
            episode_length = 0
            max_steps = 50
            
            while episode_length < max_steps:
                # 执行动作
                next_state, reward, done, info = env.step(action)
                
                # 选择下一个动作
                if not done:
                    next_valid_actions = env.get_valid_actions()
                    next_action = agent.select_action(next_state, next_valid_actions)
                else:
                    next_action = None
                
                # SARSA更新
                if next_action is not None:
                    agent.update_q_value(state, action, reward, next_state, next_action)
                
                # 更新状态和统计
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
                
                # 转移到下一个状态-动作对
                state = next_state
                # 保证 action 始终为 int 类型
                if next_action is not None:
                    action = next_action
                else:
                    # 使用等待动作作为默认动作
                    action = env.action_encoder.action_space_size - 1
            
            # 完成episode
            agent.finish_episode()
            
            # 记录统计数据
            stats['episode_rewards'].append(episode_reward)
            stats['episode_lengths'].append(episode_length)
            stats['makespans'].append(env.current_time)
            stats['completed_jobs'].append(len(env.completed_jobs))
            stats['dispatched_jobs'].append(len(env.dispatched_jobs))
            stats['epsilon'].append(agent.epsilon)
            stats['q_table_size'].append(len(agent.q_table))
            
        except Exception as e:
            print(f"Episode {episode + 1} 出错: {e}")
            # 记录错误episode的默认值
            stats['episode_rewards'].append(-100)
            stats['episode_lengths'].append(max_steps)
            stats['makespans'].append(150)
            stats['completed_jobs'].append(0)
            stats['dispatched_jobs'].append(0)
            stats['epsilon'].append(agent.epsilon)
            stats['q_table_size'].append(len(agent.q_table))
        
        # 打印进度
        if (episode + 1) % 20 == 0:
            try:
                avg_reward = np.mean(stats['episode_rewards'][-20:])
                avg_makespan = np.mean(stats['makespans'][-20:])
                print(f"Episode {episode + 1}/{episodes}")
                print(f"  平均奖励: {avg_reward:.2f}")
                print(f"  平均makespan: {avg_makespan:.2f}")
                print(f"  完成作业: {len(env.completed_jobs)}/{len(env.jobs)}")
                print(f"  派遣作业: {len(env.dispatched_jobs)}/{len(env.jobs)}")
                print(f"  Epsilon: {agent.epsilon:.3f}")
                print(f"  Q表大小: {len(agent.q_table)}")
            except Exception as e:
                print(f"打印进度时出错: {e}")
    
    total_time = time.time() - start_time
    
    # 保存结果
    try:
        result_data = {
            'stats': stats,
            'config': {
                'episodes': episodes,
                'num_jobs': len(scenario.jobs),
                'num_machines': len(scenario.machines),
                'num_distributors': len(scenario.distributors),
                'state_space_size': env.state_encoder.state_space_size,
                'action_space_size': env.action_encoder.action_space_size
            }
        }
        
        with open('sarsa_results.pkl', 'wb') as f:
            pickle.dump(result_data, f)
        
        # 保存Q表
        agent.save_model('sarsa_model.pkl')
        
        print(f"\n训练完成！")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"平均奖励: {np.mean(stats['episode_rewards']):.2f}")
        print(f"平均makespan: {np.mean(stats['makespans']):.2f}")
        print(f"最终Q表大小: {len(agent.q_table)}")
        print(f"总Q值更新次数: {agent.total_updates}")
        print(f"结果已保存至: sarsa_results.pkl")
        print(f"模型已保存至: sarsa_model.pkl")
        
    except Exception as e:
        print(f"保存结果时出错: {e}")


def test_trained_model():
    """测试训练好的模型"""
    try:
        config = Config()
        config.num_jobs = 3
        config.num_machines = 2
        config.num_distributors = 2
        
        scenario = FlexibleJobShopScenario(config=config)
        env = FJSSPEnvironment(scenario)
        
        agent = SARSAAgent(
            env.state_encoder.state_space_size,
            env.action_encoder.action_space_size,
            config
        )
        
        agent.load_model('sarsa_model.pkl')
        agent.epsilon = 0
        
        print("测试训练好的SARSA模型...")
        
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < 50:
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions)
            next_state, reward, done, info = env.step(action)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        print(f"测试结果:")
        print(f"  总奖励: {total_reward:.2f}")
        print(f"  总步数: {steps}")
        print(f"  Makespan: {env.current_time}")
        print(f"  完成作业: {len(env.completed_jobs)}/{len(env.jobs)}")
        print(f"  派遣作业: {len(env.dispatched_jobs)}/{len(env.jobs)}")
    
    except Exception as e:
        print(f"测试模型时出错: {e}")


if __name__ == "__main__":
    main()
    
    # 取消注释以测试训练好的模型
    # test_trained_model()