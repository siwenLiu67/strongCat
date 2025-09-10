import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Dict, List, Tuple, Any
import os
import sys
import time

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from icot_code.comparison_algorithms.base_algorithm import BaseAlgorithm
from icot_code.models.integrated_production_env import IntegratedFJSPEnv
from icot_code.models.transportation_model import plan_transportation   
class DQNNetwork(nn.Module):
    """DQN神经网络"""
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class DQNAlgorithm(BaseAlgorithm):
    """
    基于DQN的单层强化学习算法
    使用深度Q网络学习生产调度策略
    """
    def __init__(self, hidden_size: int = 128, buffer_size: int = 10000, 
                 batch_size: int = 64, gamma: float = 0.99, lr: float = 0.001):
        super().__init__("DQN_Algorithm")
        self.hidden_size = hidden_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        
        self.replay_buffer = deque(maxlen=buffer_size)
        self.q_network = None
        self.target_network = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.update_frequency = 10
        self.target_update_frequency = 100
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def solve(self, production_data: Dict, transport_data: Dict, orders_data: Dict, num_episodes: int = 5) -> Dict:
        """
        使用DQN算法求解生产调度问题。
        兼容统一接口：production_data, orders_data, T_internal
        """
        
        # 创建环境实例
        env = IntegratedFJSPEnv(production_data, orders_data, transport_data)
        
        if self.q_network is None:
            # 修复：直接从env实例获取状态和动作空间大小
            state_size = env.observation_space.shape[0]
            action_size = env.action_space.n
            
            self.q_network = DQNNetwork(state_size, action_size, self.hidden_size)
            self.target_network = DQNNetwork(state_size, action_size, self.hidden_size)
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        
        all_rewards = []
        start_time = time.time()
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            step = 0
            while not done:
                action = self._select_action(state)
                next_state, reward, done, _ = env.step(action)
                self.replay_buffer.append((state, action, reward, next_state, done))
                state = next_state
                episode_reward += reward
                step += 1
                if len(self.replay_buffer) >= self.batch_size and step % self.update_frequency == 0:
                    self._train_network()
                if step % self.target_update_frequency == 0:
                    if self.target_network is not None and self.q_network is not None:
                        self.target_network.load_state_dict(self.q_network.state_dict())
            all_rewards.append(episode_reward)
            print(f"Episode: {episode}, Reward: {episode_reward:.2f}, Epsilon: {self.epsilon:.2f}")

        final_state = env.reset()
        done = False
        while not done:
            action = self._select_best_action(final_state)
            next_state, _, done, info = env.step(action)
            final_state = next_state

        # 由于新环境已经集成了运输可行性检查，直接从环境中获取结果
        end_time = time.time()
        computation_time = end_time - start_time
        
        # b. 确定性运输规划
        print(f"  - 开始确定性运输规划 (生产完成时间={env.C_max})...")
        c_transport, s_trans = plan_transportation(env.C_max, transport_data['transport_data'], transport_data['orders'])
        
        transport_feasible = c_transport < float('inf')
        penalty_cost = 0.0 if transport_feasible else 1e6  #    
        print(f"  - 运输规划完成 (运输成本={c_transport}).")

        if c_transport == float('inf'):
            print(f"  - 对于生产完成时间={env.C_max} 没有可行的运输计划。返回无限大成本。")

        # c. 评估总成本
        c_production = env.C_max * production_data['c_unit_production']
        total_cost = c_production + c_transport + penalty_cost
        print(f"  - C_max: {env.C_max}, 生产成本: {c_production}, 运输成本: {c_transport}, 惩罚成本: {penalty_cost}, 总成本: {total_cost}")

        metrics = {
            "total_cost": total_cost,
            "production_cost": c_production,
            "transportation_cost": c_transport,
            "computation_time": computation_time,
            "transport_feasible": transport_feasible
        }

        result = {
            'schedule': env.schedule,
            'metrics': metrics,
            'algorithm': self.name,
            'extra_info': {'reward_curve': all_rewards}
        }
        self.results = result
        return result
    
    def _select_action(self, state: np.ndarray) -> int:
        """根据ε-贪心策略选择动作"""
        if self.q_network is None:
            raise ValueError("q_network is not initialized. Please initialize before calling _select_action.")
        if random.random() < self.epsilon:
            return random.randint(0, self.q_network.fc3.out_features - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()

    def _select_best_action(self, state: np.ndarray) -> int:
        """选择最佳动作（用于最终评估，不使用ε-贪心）"""
        if self.q_network is None:
            raise ValueError("q_network is not initialized. Please initialize before calling _select_best_action.")
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def _train_network(self) -> None:
        """从回放缓冲区中采样并训练网络"""
        if len(self.replay_buffer) < self.batch_size:
            return

        if self.q_network is None or self.target_network is None or self.optimizer is None:
            raise ValueError("Networks and optimizer must be initialized before training.")

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        current_q_values = self.q_network(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _get_state_size(self, production_data: Dict) -> int:
        num_machines = len(production_data['machines'])
        num_jobs = len(production_data['jobs'])
        return num_machines * 3 + num_jobs * 4 + 2

    def _get_action_size(self, production_data: Dict) -> int:
        num_machines = len(production_data['machines'])
        max_ops_per_machine = 10
        return num_machines * max_ops_per_machine
