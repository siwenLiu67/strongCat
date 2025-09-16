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

class ActorCriticNetwork(nn.Module):
    """PPO算法的Actor-Critic网络"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(ActorCriticNetwork, self).__init__()
        # 共享的特征提取层
        self.shared_fc1 = nn.Linear(state_size, hidden_size)
        self.shared_fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Actor网络（策略网络）
        self.actor_fc = nn.Linear(hidden_size, action_size)
        
        # Critic网络（价值网络）
        self.critic_fc = nn.Linear(hidden_size, 1)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = self.relu(self.shared_fc1(x))
        x = self.relu(self.shared_fc2(x))
        
        # Actor输出动作概率
        action_probs = self.softmax(self.actor_fc(x))
        
        # Critic输出状态价值
        state_value = self.critic_fc(x)
        
        return action_probs, state_value

class PPOAlgorithm(BaseAlgorithm):
    """
    基于PPO的单层强化学习算法
    使用近端策略优化算法学习生产调度策略
    """
    
    def __init__(self, hidden_size: int = 128, lr: float = 0.001, 
                 gamma: float = 0.99, clip_epsilon: float = 0.2):
        super().__init__("PPO_Algorithm")
        self.hidden_size = hidden_size
        self.lr = lr
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        
        # 网络将在第一次调用solve时初始化
        self.policy_network = None
        self.optimizer = None
        
        # 训练参数
        self.epochs = 3
        self.batch_size = 64
    
    def solve(self, production_data: Dict, transport_data: Dict, orders_data: Dict, num_episodes: int = 5) -> Dict:
        """
        使用PPO算法求解生产调度问题。
        兼容统一接口：production_data, transport_data, orders_data, num_episodes
        """
        
        # 创建环境实例
        env = IntegratedFJSPEnv(production_data, orders_data, transport_data)
        
        if self.policy_network is None:
            # 直接从env实例获取状态和动作空间大小
            state_size = env.observation_space.shape[0]
            action_size = env.action_space.n
            
            self.policy_network = ActorCriticNetwork(state_size, action_size, self.hidden_size)
            self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.lr)
        
        all_rewards = []
        start_time = time.time()
        
        # PPO训练循环
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            # 存储轨迹数据
            states, actions, rewards, log_probs, values = [], [], [], [], []
            
            while not done:
                # 选择动作
                action_probs, state_value = self.policy_network(torch.FloatTensor(state).unsqueeze(0))
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
                
                # 执行动作
                next_state, reward, done, _ = env.step(action.item())
                
                # 存储轨迹
                states.append(state)
                actions.append(action.item())
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(state_value.item())
                
                state = next_state
                episode_reward += reward
                step += 1
            
            # PPO更新
            if len(states) > 0:
                self._update_ppo(states, actions, rewards, log_probs, values, done)
            
            all_rewards.append(episode_reward)
            print(f"Episode: {episode}, Reward: {episode_reward:.2f}")

        # 最终评估
        final_state = env.reset()
        done = False
        while not done:
            action_probs, _ = self.policy_network(torch.FloatTensor(final_state).unsqueeze(0))
            action = torch.argmax(action_probs).item()
            next_state, _, done, info = env.step(action)
            final_state = next_state

        # 运输规划
        end_time = time.time()
        computation_time = end_time - start_time
        
        c_transport, s_trans = plan_transportation(env.C_max, production_data=production_data,
                                                   orders=transport_data['orders'],
                                                    transport_data= transport_data)
        
        transport_feasible = c_transport < float('inf')
        penalty_cost = 0.0 if transport_feasible else 1e6
        print(f"  - 运输规划完成 (运输成本={c_transport}).")

        if c_transport == float('inf'):
            print(f"  - 对于生产完成时间={env.C_max} 没有可行的运输计划。返回无限大成本。")

        # 计算总成本
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
    
    def _update_ppo(self, states, actions, rewards, log_probs, values, done):
        """PPO算法更新"""
        if len(states) == 0 or self.optimizer is None:
            return
        
        # 计算优势函数
        returns = self._compute_returns(rewards, self.gamma)
        advantages = self._compute_advantages(returns, values)
        
        # 转换为张量
        states_tensor = torch.FloatTensor(np.array(states))
        actions_tensor = torch.LongTensor(actions)
        old_log_probs_tensor = torch.stack(log_probs)
        returns_tensor = torch.FloatTensor(returns)
        advantages_tensor = torch.FloatTensor(advantages)
        
        # PPO多轮更新
        for _ in range(self.epochs):
            # 获取新的动作概率和价值
            action_probs, state_values = self.policy_network(states_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            new_log_probs = action_dist.log_prob(actions_tensor)
            
            # 计算概率比
            ratio = torch.exp(new_log_probs - old_log_probs_tensor.detach())
            
            # 计算裁剪的PPO损失
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_tensor
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 价值函数损失
            value_loss = nn.MSELoss()(state_values.squeeze(), returns_tensor)
            
            # 熵奖励
            entropy = action_dist.entropy().mean()
            
            # 总损失
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def _compute_returns(self, rewards, gamma):
        """计算回报"""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return returns
    
    def _compute_advantages(self, returns, values):
        """计算优势函数"""
        advantages = []
        for i in range(len(returns)):
            advantage = returns[i] - values[i]
            advantages.append(advantage)
        return advantages
    
    def _get_state_size(self, production_data: Dict) -> int:
        """获取状态空间大小"""
        num_machines = len(production_data['machines'])
        num_jobs = len(production_data['jobs'])
        return num_machines * 3 + num_jobs * 4 + 2
    
    def _get_action_size(self, production_data: Dict) -> int:
        """获取动作空间大小"""
        num_machines = len(production_data['machines'])
        max_ops_per_machine = 10
        return num_machines * max_ops_per_machine
