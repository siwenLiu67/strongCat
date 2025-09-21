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
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super(ActorCriticNetwork, self).__init__()
        # 共享的特征提取层
        self.shared_fc1 = nn.Linear(state_size, hidden_size)
        self.shared_fc2 = nn.Linear(hidden_size, hidden_size)
        self.shared_fc3 = nn.Linear(hidden_size, hidden_size)
        
        # Actor网络（策略网络）
        self.actor_fc = nn.Linear(hidden_size, action_size)
        
        # Critic网络（价值网络）
        self.critic_fc = nn.Linear(hidden_size, 1)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module == self.actor_fc:
                torch.nn.init.orthogonal_(module.weight, gain=0.01)
            else:
                torch.nn.init.orthogonal_(module.weight, gain=1.0)
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        x = self.tanh(self.shared_fc1(x))
        x = self.tanh(self.shared_fc2(x))
        x = self.tanh(self.shared_fc3(x))
        
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
    
    def __init__(self, hidden_size: int = 256, lr: float = 3e-4, 
                 gamma: float = 0.99, clip_epsilon: float = 0.2, gae_lambda: float = 0.95):
        super().__init__("PPO_Algorithm")
        self.hidden_size = hidden_size
        self.lr = lr
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.gae_lambda = gae_lambda
        
        # 网络将在第一次调用solve时初始化
        self.policy_network = None
        self.optimizer = None
        self.scheduler = None
        
        # 训练参数
        self.epochs = 10
        self.batch_size = 64
        
        # 奖励标准化参数
        self.reward_mean = 0
        self.reward_std = 1
        self.reward_count = 1e-4
        
        # 价值函数标准化参数
        self.value_mean = 0
        self.value_std = 1
        self.value_count = 1e-4
    
    def solve(self, production_data: Dict, transport_data: Dict, orders_data: Dict, num_episodes: int = 500) -> Dict:
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
            self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.lr, eps=1e-5)
            self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=num_episodes)
        
        all_rewards = []
        start_time = time.time()
        
        # PPO训练循环
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            # 存储轨迹数据
            states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
            
            while not done:
                # 选择动作
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_probs, state_value = self.policy_network(state_tensor)
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
                
                # 执行动作
                next_state, reward, done, _ = env.step(action.item())
                
                # 更新奖励标准化参数
                self._update_reward_stats(reward)
                
                # 标准化奖励
                normalized_reward = (reward - self.reward_mean) / (self.reward_std + 1e-8)
                
                # 存储轨迹
                states.append(state)
                actions.append(action.item())
                rewards.append(normalized_reward)
                log_probs.append(log_prob)
                values.append(state_value.item())
                dones.append(done)
                
                state = next_state
                episode_reward += reward
                step += 1
            
            # 每集结束后进行PPO更新
            if len(states) > 0:
                self._update_ppo(states, actions, rewards, log_probs, values, dones)
            
            # 更新学习率
            self.scheduler.step()
            
            all_rewards.append(episode_reward)
            print(f"Episode: {episode}, Reward: {episode_reward:.2f}, Steps: {step}")

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
    
    def _update_reward_stats(self, reward):
        """更新奖励统计信息用于标准化"""
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = reward - self.reward_mean
        self.reward_std += delta * delta2
    
    def _update_ppo(self, states, actions, rewards, log_probs, values, dones):
        """PPO算法更新"""
        if len(states) == 0 or self.optimizer is None:
            return

        # 计算GAE优势函数
        advantages = self._compute_gae(rewards, values, dones, self.gamma, self.gae_lambda)
        
        # 计算回报
        returns = advantages + np.array(values)
        
        # 更新价值函数标准化参数
        self._update_value_stats(returns)
        
        # 标准化advantages和returns
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        returns = (returns - self.value_mean) / (self.value_std + 1e-8)

        # 转换为张量
        states_tensor = torch.FloatTensor(np.array(states))
        actions_tensor = torch.LongTensor(actions)
        old_log_probs_tensor = torch.FloatTensor([lp.item() for lp in log_probs])
        returns_tensor = torch.FloatTensor(returns)
        advantages_tensor = torch.FloatTensor(advantages)
        old_values_tensor = torch.FloatTensor(values)

        # PPO多轮更新
        for _ in range(self.epochs):
            # 随机打乱数据
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            # 小批量更新
            for start in range(0, len(states), self.batch_size):
                end = min(start + self.batch_size, len(states))
                batch_indices = indices[start:end]
                
                if len(batch_indices) == 0:
                    continue
                
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_old_values = old_values_tensor[batch_indices]
                
                # 前向传播
                action_probs, state_values = self.policy_network(batch_states)
                action_dist = torch.distributions.Categorical(action_probs)
                new_log_probs = action_dist.log_prob(batch_actions)
                
                # 计算比率和策略损失
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值函数损失（带裁剪）
                value_pred_clipped = batch_old_values + torch.clamp(
                    state_values.squeeze() - batch_old_values, -self.clip_epsilon, self.clip_epsilon
                )
                value_loss1 = (state_values.squeeze() - batch_returns).pow(2)
                value_loss2 = (value_pred_clipped - batch_returns).pow(2)
                value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                
                # 熵奖励
                entropy = action_dist.entropy().mean()
                
                # 总损失
                loss = policy_loss + value_loss - 0.01 * entropy
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=0.5)
                
                self.optimizer.step()
                
                # 打印调试信息
                if start == 0 and _ == 0:
                    print(f"[PPO DEBUG] rewards: mean={np.mean(rewards):.4f}, std={np.std(rewards):.4f}, "
                          f"min={np.min(rewards):.4f}, max={np.max(rewards):.4f}")
                    print(f"[PPO DEBUG] advantages: mean={np.mean(advantages):.4f}, std={np.std(advantages):.4f}, "
                          f"min={np.min(advantages):.4f}, max={np.max(advantages):.4f}")
                    
                    # 检查参数更新
                    for name, param in self.policy_network.named_parameters():
                        if "weight" in name and "shared_fc1" in name:
                            print(f"[PPO DEBUG] After update, {name} mean: {param.data.mean():.6f}, std: {param.data.std():.6f}")
                            break
    
    def _update_value_stats(self, values):
        """更新价值函数统计信息用于标准化"""
        batch_mean = np.mean(values)
        batch_std = np.std(values)
        batch_count = len(values)
        
        # 更新运行统计
        total_count = self.value_count + batch_count
        delta = batch_mean - self.value_mean
        new_mean = self.value_mean + delta * batch_count / total_count
        
        # 更新标准差
        m_a = self.value_std * self.value_count
        m_b = batch_std * batch_count
        M2 = m_a + m_b + delta**2 * self.value_count * batch_count / total_count
        new_std = np.sqrt(M2 / total_count)
        
        self.value_mean = new_mean
        self.value_std = new_std
        self.value_count = total_count
    
    def _compute_gae(self, rewards, values, dones, gamma, gae_lambda):
        """使用GAE计算优势函数"""
        advantages = []
        last_advantage = 0
        next_value = 0
        
        # 反向计算
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - int(dones[t])
                next_value = values[t]  # 使用当前值作为下一个状态的估计
            else:
                next_non_terminal = 1.0 - int(dones[t+1])
                next_value = values[t+1]
            
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
            advantages.insert(0, last_advantage)
        
        return np.array(advantages)
    
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