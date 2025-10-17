"""
使用PPO（Proximal Policy Optimization）算法解决集成柔性作业车间调度与派遣问题(IFJSSP-DP)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import random
import pickle
import time
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
from ppo_environment_adaptor import FJSSPEnvironment
from case_generator import FlexibleJobShopScenario
from config import Config
from data_structures import Job, Operation, Machine


class PPOActorCritic(nn.Module):
    """PPO Actor-Critic网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(PPOActorCritic, self).__init__()
        
        # 共享特征层
        self.shared_fc1 = nn.Linear(state_dim, hidden_dim)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Actor网络（策略网络）
        self.actor_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.actor_fc2 = nn.Linear(hidden_dim, action_dim)
        
        # Critic网络（价值网络）
        self.critic_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.critic_fc2 = nn.Linear(hidden_dim, 1)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, state):
        # 共享特征提取
        x = F.relu(self.shared_fc1(state))
        x = self.dropout(x)
        x = F.relu(self.shared_fc2(x))
        x = self.dropout(x)
        
        # Actor输出（动作概率）
        actor_x = F.relu(self.actor_fc1(x))
        action_logits = self.actor_fc2(actor_x)
        
        # Critic输出（状态价值）
        critic_x = F.relu(self.critic_fc1(x))
        state_value = self.critic_fc2(critic_x)
        
        return action_logits, state_value
    
    def get_action_and_value(self, state, valid_actions_mask=None):
        """获取动作和价值"""
        action_logits, state_value = self.forward(state)
        
        # 如果有有效动作掩码，则屏蔽无效动作
        if valid_actions_mask is not None:
            action_logits = action_logits.masked_fill(~valid_actions_mask, float('-inf'))
        
        # 创建动作分布
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        
        # 采样动作
        action = dist.sample()
        
        return action, dist.log_prob(action), dist.entropy(), state_value
    
    def get_value(self, state):
        """仅获取状态价值"""
        _, state_value = self.forward(state)
        return state_value
    
    def evaluate_actions(self, state, action, valid_actions_mask=None):
        """评估给定的动作"""
        action_logits, state_value = self.forward(state)
        
        if valid_actions_mask is not None:
            action_logits = action_logits.masked_fill(~valid_actions_mask, float('-inf'))
        
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        
        return dist.log_prob(action), dist.entropy(), state_value


class PPOBuffer:
    """PPO经验缓冲区"""
    
    def __init__(self, capacity: int, state_dim: int, gamma: float = 0.99, lam: float = 0.95):
        self.capacity = capacity
        self.gamma = gamma
        self.lam = lam
        
        # 缓冲区
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)
        
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = capacity
    
    def store(self, state, action, reward, value, log_prob, done):
        """存储一步经验"""
        assert self.ptr < self.max_size
        
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        self.ptr += 1
    
    def finish_path(self, last_value=0):
        """完成一个轨迹，计算优势和回报"""
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.rewards[path_slice], last_value)
        values = np.append(self.values[path_slice], last_value)
        
        # GAE-Lambda优势估计
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.advantages[path_slice] = self._discount_cumsum(deltas, self.gamma * self.lam)
        
        # 计算回报
        self.returns[path_slice] = self._discount_cumsum(rewards, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr
    
    def get(self):
        """获取所有数据并重置缓冲区"""
        assert self.ptr == self.max_size
        
        # 标准化优势
        adv_mean = np.mean(self.advantages)
        adv_std = np.std(self.advantages)
        self.advantages = (self.advantages - adv_mean) / (adv_std + 1e-8)
        
        data = dict(
            states=self.states,
            actions=self.actions,
            returns=self.returns,
            advantages=self.advantages,
            log_probs=self.log_probs
        )
        
        # 重置
        self.ptr = 0
        self.path_start_idx = 0
        
        return data
    
    def _discount_cumsum(self, x, discount):
        """计算折扣累积和"""
        cumsum = np.zeros_like(x)
        cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            cumsum[t] = x[t] + discount * cumsum[t + 1]
        return cumsum




class PPOAgent:
    """PPO智能体"""
    
    def __init__(self, state_dim: int, action_dim: int, config: Config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # PPO超参数
        self.lr = 3e-4
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_ratio = 0.2
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.max_grad_norm = 0.5
        self.ppo_epochs = 10
        self.buffer_size = 2048
        self.batch_size = 64
        
        # 网络
        self.actor_critic = PPOActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.lr)
        
        # 经验缓冲区
        self.buffer = PPOBuffer(self.buffer_size, state_dim, self.gamma, self.lam)
        
        # 统计
        self.update_count = 0
    
    def select_action(self, state: np.ndarray, action_mask: Optional[np.ndarray] = None) -> Tuple[int, float, float]:
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0) if action_mask is not None else None
        
        with torch.no_grad():
            action, log_prob, entropy, value = self.actor_critic.get_action_and_value(
                state_tensor, mask_tensor
            )
        
        return int(action.item()), float(log_prob.item()), float(value.item())
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        """存储转换"""
        self.buffer.store(state, action, reward, value, log_prob, done)
    
    def finish_path(self, last_value=0):
        """完成路径"""
        self.buffer.finish_path(last_value)
    
    def update(self):
        """更新网络"""
        if self.buffer.ptr < self.buffer_size:
            return {}
        
        # 获取数据
        data = self.buffer.get()
        
        # 转换为tensor
        states = torch.FloatTensor(data['states'])
        actions = torch.LongTensor(data['actions'])
        returns = torch.FloatTensor(data['returns'])
        advantages = torch.FloatTensor(data['advantages'])
        old_log_probs = torch.FloatTensor(data['log_probs'])
        
        # 多轮更新
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
        for _ in range(self.ppo_epochs):
            # 随机打乱数据
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # 前向传播
                log_probs, entropy, values = self.actor_critic.evaluate_actions(
                    batch_states, batch_actions
                )
                
                # 计算比率
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # PPO损失
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # 熵损失
                entropy_loss = -entropy.mean()
                
                # 总损失
                total_loss = (policy_loss + 
                            self.value_coef * value_loss + 
                            self.entropy_coef * entropy_loss)
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
        
        self.update_count += 1
        
        return {
            'policy_loss': total_policy_loss / self.ppo_epochs,
            'value_loss': total_value_loss / self.ppo_epochs,
            'entropy_loss': total_entropy_loss / self.ppo_epochs
        }


def main():
    """主训练函数"""
    # 配置参数
    config = Config()
    config.num_initial_jobs = 6
    config.num_machines = 3
    config.num_distributors = 2
    config.min_operations = 2
    config.max_operations = 3
    
    print("生成FJSP-DP场景...")
    scenario = FlexibleJobShopScenario(config=config)
    
    print(f"场景信息:")
    print(f"- 作业数量: {len(scenario.jobs)}")
    print(f"- 机器数量: {len(scenario.machines)}")
    print(f"- 配送商数量: {len(scenario.distributors)}")
    
    # 创建环境和智能体
    env = FJSSPEnvironment(scenario)
    state_dim = 20
    action_dim = env.action_dim
    
    agent = PPOAgent(state_dim, action_dim, config)
    
    # 训练参数
    episodes = 100
    stats = defaultdict(list)
    
    print(f"\n开始PPO训练 {episodes} episodes...")
    start_time = time.time()
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            # 获取动作掩码
            action_mask = env.get_action_mask()
            
            # 选择动作
            action, log_prob, value = agent.select_action(state, action_mask)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            agent.store_transition(state, action, reward, value, log_prob, done)
            
            # 更新状态和统计
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                # 完成路径
                last_value = agent.actor_critic.get_value(torch.FloatTensor(state).unsqueeze(0)).item()
                agent.finish_path(last_value)
                break
        
        # 记录统计数据
        stats['episode_rewards'].append(episode_reward)
        stats['episode_lengths'].append(episode_length)
        stats['makespans'].append(env.current_time)
        stats['completed_jobs'].append(len(env.completed_jobs))
        stats['dispatched_jobs'].append(len(env.dispatched_jobs))
        
        # 从环境中获取延迟惩罚和目标值指标
        stats['tardy_penalty'] = stats.get('tardy_penalty', [])
        stats['total_weighted_tardiness'] = stats.get('total_weighted_tardiness', [])
        stats['objective_value'] = stats.get('objective_value', [])
        
        # 获取环境中的惩罚指标
        tardy_penalty = env.tardy_penalty
        total_weighted_tardiness = env.total_weighted_tardiness
        
        stats['tardy_penalty'].append(tardy_penalty)
        stats['total_weighted_tardiness'].append(total_weighted_tardiness)
        stats['objective_value'].append(episode_reward + tardy_penalty + total_weighted_tardiness)
        
        # 更新网络
        if agent.buffer.ptr >= agent.buffer_size:
            update_info = agent.update()
            if update_info:
                stats['policy_loss'].append(update_info['policy_loss'])
                stats['value_loss'].append(update_info['value_loss'])
                stats['entropy_loss'].append(update_info['entropy_loss'])
        
        # 打印进度
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(stats['episode_rewards'][-10:])
            avg_makespan = np.mean(stats['makespans'][-10:])
            print(f"Episode {episode + 1}/{episodes}")
            print(f"  平均奖励: {avg_reward:.2f}")
            print(f"  平均makespan: {avg_makespan:.2f}")
            print(f"  完成作业: {len(env.completed_jobs)}/{len(env.jobs)}")
            print(f"  派遣作业: {len(env.dispatched_jobs)}/{len(env.jobs)}")
            
            if stats['policy_loss']:
                print(f"  策略损失: {stats['policy_loss'][-1]:.4f}")
                print(f"  价值损失: {stats['value_loss'][-1]:.4f}")
    
    total_time = time.time() - start_time
    
    # 保存结果
    result_data = {
        'stats': stats,
        'config': {
            'episodes': episodes,
            'num_jobs': len(scenario.jobs),
            'num_machines': len(scenario.machines),
            'num_distributors': len(scenario.distributors)
        }
    }
    
    with open('ppo_results.pkl', 'wb') as f:
        pickle.dump(result_data, f)
    
    # 保存模型
    torch.save(agent.actor_critic.state_dict(), 'ppo_model.pth')
    
    print(f"\n训练完成！")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"平均奖励: {np.mean(stats['episode_rewards']):.2f}")
    print(f"平均makespan: {np.mean(stats['makespans']):.2f}")
    print(f"平均完成作业数: {np.mean(stats['completed_jobs']):.2f}")
    print(f"平均派遣作业数: {np.mean(stats['dispatched_jobs']):.2f}")
    print(f"结果已保存至: ppo_results.pkl")
    print(f"模型已保存至: ppo_model.pth")


def test_trained_model():
    """测试训练好的模型"""
    config = Config()
    config.num_initial_jobs = 5
    config.num_machines = 3
    config.num_distributors = 2
    
    # 生成测试场景
    scenario = FlexibleJobShopScenario(config=config)
    env = FJSSPEnvironment(scenario)
    
    # 加载训练好的模型
    state_dim = 20
    action_dim = env.action_dim
    agent = PPOAgent(state_dim, action_dim, config)
    agent.actor_critic.load_state_dict(torch.load('ppo_model.pth'))
    
    print("测试训练好的PPO模型...")
    
    state = env.reset()
    total_reward = 0
    
    while True:
        action_mask = env.get_action_mask()
        action, _, _ = agent.select_action(state, action_mask)
        next_state, reward, done, info = env.step(action)
        
        total_reward += reward
        state = next_state
        
        if done:
            break
    
    print(f"测试结果:")
    print(f"  总奖励: {total_reward:.2f}")
    print(f"  Makespan: {env.current_time}")
    print(f"  完成作业: {len(env.completed_jobs)}/{len(env.jobs)}")
    print(f"  派遣作业: {len(env.dispatched_jobs)}/{len(env.jobs)}")


def run_ppo_experiment(config, case, seed, **kwargs):
    """
    批量实验统一入口，供批量运行器调用
    """
    # 设置随机种子
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 构建环境和智能体
    env = FJSSPEnvironment(case)
    state_dim = 20
    action_dim = env.action_dim
    agent = PPOAgent(state_dim, action_dim, config)
    episodes = config.episodes
    stats = defaultdict(list)

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            # 获取动作掩码
            action_mask = env.get_action_mask()
            
            # 选择动作
            action, log_prob, value = agent.select_action(state, action_mask)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            agent.store_transition(state, action, reward, value, log_prob, done)
            
            # 更新状态和统计
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                # 完成路径
                last_value = agent.actor_critic.get_value(torch.FloatTensor(state).unsqueeze(0)).item()
                agent.finish_path(last_value)
                break
        
        # 记录统计数据
        stats['episode_rewards'].append(episode_reward)
        stats['episode_lengths'].append(episode_length)
        stats['makespans'].append(env.current_time)
        stats['completed_jobs'].append(len(env.completed_jobs))
        stats['dispatched_jobs'].append(len(env.dispatched_jobs))
        
        
        # 获取环境中的惩罚指标
        # tardy_penalty = getattr(env, 'tardy_penalty', 0)
        # total_weighted_tardiness = getattr(env, 'total_weighted_tardiness', 0)
        tardy_penalty = env.tardy_penalty 
        total_weighted_tardiness = env.total_weighted_tardiness
        
        stats['tardy_penalty'].append(tardy_penalty)
        stats['total_weighted_tardiness'].append(total_weighted_tardiness)
        stats['objective_value'].append(episode_reward + tardy_penalty + total_weighted_tardiness)
        
        # 更新网络
        if agent.buffer.ptr >= agent.buffer_size:
            update_info = agent.update()
            if update_info:
                stats['policy_loss'].append(update_info['policy_loss'])
                stats['value_loss'].append(update_info['value_loss'])
                stats['entropy_loss'].append(update_info['entropy_loss'])

    result = {
        "stats": stats,
        "env": env,
        "additional_metrics": {}
    }
    return result


if __name__ == "__main__":
    main()
    
    # 取消注释以测试训练好的模型
    # test_trained_model()
