import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from collections import deque
import random

class HighLevelController(nn.Module):
    """高层控制器 - 负责设定子目标和在更长的时间尺度上决策"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.subgoal_dim = 4  # 子目标维度：完成作业数、机器利用率、配送完成度、时间进度
        self.action_dim = 3   # 动作空间：调度、配送、等待
        
        # 网络结构
        self.state_dim = 12   # 状态特征维度
        self.hidden_dim = 64
        
        self.fc1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.action_dim)
        
        # 目标网络
        self.target_net = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim)
        )
        self.target_net.load_state_dict(self.state_dict())
        
        # 训练参数
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update_freq = 100
        self.step_count = 0
        
        # 经验回放
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 32
        
        # 子目标跟踪
        self.current_subgoal = None
        self.subgoal_duration = 0
        self.max_subgoal_duration = 20  # 子目标最大持续时间
        
    def _extract_state_features(self, state: Dict) -> torch.Tensor:
        """提取高层状态特征"""
        jobs = state['available_jobs']
        completed_jobs = state.get('completed_jobs', [])
        dispatched_jobs = state.get('dispatched_jobs', [])
        machines = state['machines']
        current_time = state.get('current_time', 0)
        
        # 计算关键指标
        total_jobs = len(jobs) + len(completed_jobs) + len(dispatched_jobs)
        completion_rate = len(completed_jobs) / max(1, total_jobs)
        dispatch_rate = len(dispatched_jobs) / max(1, total_jobs)
        
        # 机器利用率
        busy_machines = sum(1 for m in machines if m.status == 'busy')
        machine_utilization = busy_machines / len(machines) if machines else 0
        
        # 延误情况
        tardiness = sum(max(0, getattr(j, 'dispatched_time', current_time) - getattr(j, 'due_date', current_time)) 
                       for j in completed_jobs + dispatched_jobs)
        max_tardiness = max([getattr(j, 'due_date', 1) for j in completed_jobs + dispatched_jobs] + [1])
        normalized_tardiness = tardiness / max_tardiness
        
        # 配送满足度
        delivery_satisfaction = self._calculate_delivery_satisfaction(state)
        
        # 时间进度
        time_progress = current_time / self.config.max_time_steps
        
        # 作业等待时间
        avg_wait_time = np.mean([getattr(j, 'waiting_time', 0) for j in jobs]) if jobs else 0
        normalized_wait_time = avg_wait_time / max(1, self.config.max_processing_time)
        
        # 构建状态特征向量
        features = torch.tensor([
            completion_rate,           # 完成率
            dispatch_rate,             # 配送率
            machine_utilization,       # 机器利用率
            normalized_tardiness,      # 归一化延误
            delivery_satisfaction,     # 配送满足度
            time_progress,             # 时间进度
            normalized_wait_time,      # 归一化等待时间
            len(jobs) / max(1, total_jobs),  # 待处理作业比例
            len(completed_jobs) / max(1, total_jobs),  # 已完成作业比例
            len(dispatched_jobs) / max(1, total_jobs), # 已配送作业比例
            busy_machines / max(1, len(machines)),     # 忙碌机器比例
            current_time / max(1, self.config.max_time_steps)  # 时间进度
        ], dtype=torch.float32)
        
        return features.unsqueeze(0)
    
    def _calculate_delivery_satisfaction(self, state: Dict) -> float:
        """计算配送要求满足度"""
        distributors = state.get('distributors', [])
        if not distributors:
            return 0.0
            
        total_satisfaction = 0
        for distributor in distributors:
            if not hasattr(distributor, 'delivery_requirements') or not distributor.delivery_requirements:
                continue
                
            requirement = distributor.delivery_requirements
            current_time = state.get('current_time', 0)
            
            for due_time, ratio, weight in zip(requirement.due_times, requirement.ratios, requirement.weights):
                if current_time <= due_time:
                    # 时间窗口内，计算满足度
                    completed_jobs = [j for j in state.get('completed_jobs', []) 
                                    if getattr(j, 'dispatched_time', float('inf')) <= due_time]
                    completed_amount = sum(getattr(j, 'amount', 1) for j in completed_jobs)
                    required_amount = ratio * getattr(distributor, 'total_amount', 1)
                    
                    if required_amount > 0:
                        satisfaction = min(1.0, completed_amount / required_amount)
                        total_satisfaction += satisfaction * weight
        
        return total_satisfaction / len(distributors) if distributors else 0.0
    
    def select_action(self, state: Dict) -> int:
        """选择高层动作"""
        state_features = self._extract_state_features(state)
        
        # epsilon-贪婪策略
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                q_values = self.forward(state_features)
                action = q_values.argmax().item()
        
        # 更新epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return action
    
    def set_subgoal(self, state: Dict, action: int) -> Dict:
        """根据当前状态和动作设定子目标"""
        current_time = state.get('current_time', 0)
        completed_jobs = state.get('completed_jobs', [])
        dispatched_jobs = state.get('dispatched_jobs', [])
        machines = state['machines']
        
        # 根据动作类型设定不同的子目标
        if action == 0:  # 调度动作
            # 子目标：提高机器利用率或完成更多作业
            current_utilization = sum(1 for m in machines if m.status == 'busy') / len(machines)
            target_utilization = min(1.0, current_utilization + 0.2)
            
            subgoal = {
                'type': 'scheduling',
                'target_utilization': target_utilization,
                'target_completions': len(completed_jobs) + 2,  # 完成2个额外作业
                'duration': self.max_subgoal_duration
            }
            
        elif action == 1:  # 配送动作
            # 子目标：提高配送满足度
            current_satisfaction = self._calculate_delivery_satisfaction(state)
            target_satisfaction = min(1.0, current_satisfaction + 0.3)
            
            subgoal = {
                'type': 'dispatching',
                'target_satisfaction': target_satisfaction,
                'target_dispatched': len(dispatched_jobs) + 1,  # 配送1个额外作业
                'duration': self.max_subgoal_duration
            }
            
        else:  # 等待动作
            # 子目标：等待特定条件满足
            subgoal = {
                'type': 'waiting',
                'target_time': current_time + 5,  # 等待5个时间步
                'duration': 5
            }
        
        self.current_subgoal = subgoal
        self.subgoal_duration = 0
        
        return subgoal
    
    def check_subgoal_completion(self, state: Dict) -> Tuple[bool, float]:
        """检查子目标完成情况并返回内在奖励"""
        if self.current_subgoal is None:
            return False, 0.0
        
        self.subgoal_duration += 1
        
        # 检查是否超时
        if self.subgoal_duration >= self.current_subgoal['duration']:
            self.current_subgoal = None
            return True, -0.1  # 超时惩罚
        
        completion = False
        intrinsic_reward = 0.0
        
        if self.current_subgoal['type'] == 'scheduling':
            # 检查调度子目标完成情况
            machines = state['machines']
            current_utilization = sum(1 for m in machines if m.status == 'busy') / len(machines)
            completed_jobs = state.get('completed_jobs', [])
            
            util_complete = current_utilization >= self.current_subgoal['target_utilization']
            comp_complete = len(completed_jobs) >= self.current_subgoal['target_completions']
            
            if util_complete or comp_complete:
                completion = True
                intrinsic_reward = 1.0  # 子目标完成奖励
                
        elif self.current_subgoal['type'] == 'dispatching':
            # 检查配送子目标完成情况
            current_satisfaction = self._calculate_delivery_satisfaction(state)
            dispatched_jobs = state.get('dispatched_jobs', [])
            
            sat_complete = current_satisfaction >= self.current_subgoal['target_satisfaction']
            disp_complete = len(dispatched_jobs) >= self.current_subgoal['target_dispatched']
            
            if sat_complete or disp_complete:
                completion = True
                intrinsic_reward = 1.0
                
        elif self.current_subgoal['type'] == 'waiting':
            # 检查等待子目标完成情况
            current_time = state.get('current_time', 0)
            if current_time >= self.current_subgoal['target_time']:
                completion = True
                intrinsic_reward = 0.5  # 较小的等待完成奖励
        
        if completion:
            self.current_subgoal = None
        
        return completion, intrinsic_reward
    
    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = F.relu(self.fc1(state_features))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def update(self) -> float:
        """更新网络参数"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # 随机采样批次
        batch_samples = random.sample(self.replay_buffer, self.batch_size)
        
        states = torch.stack([sample[0] for sample in batch_samples])
        actions = torch.tensor([sample[1] for sample in batch_samples], dtype=torch.long)
        rewards = torch.tensor([sample[2] for sample in batch_samples], dtype=torch.float32)
        next_states = torch.stack([sample[3] for sample in batch_samples])
        dones = torch.tensor([sample[4] for sample in batch_samples], dtype=torch.float32)
        
        # 计算当前Q值
        current_q_values = self.forward(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # 计算损失
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.state_dict())
        
        return loss.item()
    
    def store_experience(self, state: Dict, action: int, reward: float, next_state: Dict, done: bool):
        """存储经验到回放缓冲区"""
        state_features = self._extract_state_features(state)
        next_state_features = self._extract_state_features(next_state)
        
        self.replay_buffer.append((
            state_features.squeeze(),
            action,
            reward,
            next_state_features.squeeze(),
            done
        ))
