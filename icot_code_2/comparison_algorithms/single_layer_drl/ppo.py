"""
近端策略优化 (Proximal Policy Optimization)
"""
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
from ..base_algorithm import BaseAlgorithm


class ActorCritic(nn.Module):
    """Actor-Critic 网络模型"""
    def __init__(self, n_obs, n_actions, hidden_dim=128):
        super(ActorCritic, self).__init__()
        
        # 共享特征提取层
        self.shared_layers = nn.Sequential(
            nn.Linear(n_obs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor 头 (策略网络)
        self.actor_head = nn.Linear(hidden_dim, n_actions)
        
        # Critic 头 (价值网络)
        self.critic_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        shared_features = self.shared_layers(x)
        action_logits = self.actor_head(shared_features)
        state_value = self.critic_head(shared_features)
        return action_logits, state_value
    
    def get_action_and_value(self, x, valid_mask, action=None):
        """获取动作和价值，支持mask"""
        logits, value = self.forward(x)
        
        # 将无效动作的logits设置为极小值
        logits = logits.clone()
        logits[valid_mask == 0] = -1e8
        
        # 创建分布
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value


class RolloutBuffer:
    """存储轨迹数据的缓冲区"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.valid_masks = []
        
    def push(self, state, action, log_prob, reward, value, done, valid_mask):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.valid_masks.append(valid_mask)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.valid_masks.clear()
    
    def get(self):
        return (self.states, self.actions, self.log_probs, 
                self.rewards, self.values, self.dones, self.valid_masks)
    
    def __len__(self):
        return len(self.rewards)


class PPO_Agent(BaseAlgorithm):
    """
    使用 PPO 求解 FJSP 的智能体。
    """
    def __init__(self, env, config):
        """
        :param env: FjspEnv, a scheduling environment instance.
        :param config: dict, a dictionary containing algorithm-specific hyperparameters.
        """
        super().__init__(env, config)
        
        # 从配置中加载超参数
        self.episodes = self.config.get('episodes', 500)
        self.gamma = self.config.get('gamma', 0.99)
        self.gae_lambda = self.config.get('gae_lambda', 0.95)
        self.lr = self.config.get('lr', 3e-4)
        self.clip_epsilon = self.config.get('clip_epsilon', 0.2)
        self.value_coef = self.config.get('value_coef', 0.5)
        self.entropy_coef = self.config.get('entropy_coef', 0.01)
        self.max_grad_norm = self.config.get('max_grad_norm', 0.5)
        self.update_epochs = self.config.get('update_epochs', 4)
        self.mini_batch_size = self.config.get('mini_batch_size', 64)
        
        # 状态维度：机器释放时间 + 每个作业的下一道工序索引 + 已完成工序比例
        self.state_dim = self.n_machines + self.n_jobs + 1
        
        # 初始化Actor-Critic网络
        self.policy = ActorCritic(self.state_dim, self.n_ops)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        
        # 初始化经验缓冲区
        self.rollout_buffer = RolloutBuffer()
        
    def _get_state(self, machine_release_times, job_next_op, scheduled_ops_count):
        """构建状态表示"""
        m_times = np.copy(machine_release_times)
        j_ops = np.copy(job_next_op)
        
        # 归一化以避免大数值
        m_times_norm = m_times / (np.mean(m_times) + 1e-9) if np.mean(m_times) > 0 else m_times
        
        # 找到最大工序数用于归一化
        max_ops_per_job = 0
        for job_spec in self.env.jobs:
            if len(job_spec['operations']) > max_ops_per_job:
                max_ops_per_job = len(job_spec['operations'])
        
        j_ops_norm = j_ops / (max_ops_per_job + 1e-9) if max_ops_per_job > 0 else j_ops
        
        progress = np.array([scheduled_ops_count / self.n_ops])
        
        state_np = np.concatenate([m_times_norm, j_ops_norm, progress]).astype(np.float32)
        return torch.from_numpy(state_np).unsqueeze(0)
    
    def _compute_gae(self, rewards, values, dones, next_value):
        """计算广义优势估计 (Generalized Advantage Estimation)"""
        advantages = []
        gae = 0
        
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[step + 1]
            
            delta = rewards[step] + self.gamma * next_val * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns
    
    def _update_policy(self):
        """使用收集的经验更新策略"""
        states, actions, old_log_probs, rewards, values, dones, valid_masks = self.rollout_buffer.get()
        
        # 计算最后一个状态的价值用于GAE
        with torch.no_grad():
            if len(states) > 0:
                last_state = states[-1]
                _, last_value = self.policy(last_state)
                last_value = last_value.item()
            else:
                last_value = 0
        
        # 提取标量值
        values_np = [v.item() for v in values]
        rewards_np = rewards
        dones_np = dones
        
        # 计算优势和回报
        advantages, returns = self._compute_gae(rewards_np, values_np, dones_np, last_value)
        
        # 转换为tensor
        states_tensor = torch.cat(states)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        old_log_probs_tensor = torch.tensor(old_log_probs, dtype=torch.float32)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        valid_masks_tensor = torch.stack(valid_masks)
        
        # 归一化优势
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # 多轮更新
        dataset_size = len(states)
        for _ in range(self.update_epochs):
            # 随机打乱数据
            indices = np.random.permutation(dataset_size)
            
            # Mini-batch更新
            for start_idx in range(0, dataset_size, self.mini_batch_size):
                end_idx = min(start_idx + self.mini_batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_valid_masks = valid_masks_tensor[batch_indices]
                
                # 前向传播
                _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(
                    batch_states, batch_valid_masks, batch_actions
                )
                
                # 计算ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO clip损失
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 价值函数损失
                value_loss = nn.MSELoss()(new_values.squeeze(), batch_returns)
                
                # 熵损失 (鼓励探索)
                entropy_loss = -entropy.mean()
                
                # 总损失
                loss = actor_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        # 清空缓冲区
        self.rollout_buffer.clear()
    
    def solve(self):
        """
        训练PPO智能体并返回最佳解决方案
        """
        print(f"Running {self.__class__.__name__} with training for {self.episodes} episodes...")
        
        best_makespan = float('inf')
        
        # --- 训练循环 ---
        for i_episode in range(self.episodes):
            # --- 重置环境 ---
            machine_release_times = np.zeros(self.n_machines)
            op_completion_times = {}
            job_next_op = np.zeros(self.n_jobs, dtype=int)
            scheduled_ops_mask = np.zeros(self.n_ops, dtype=int)
            
            last_makespan = 0
            episode_reward = 0
            
            for step in range(self.n_ops):
                # 获取当前状态
                state = self._get_state(machine_release_times, job_next_op, step)
                
                # 确定有效动作
                valid_mask = np.zeros(self.n_ops)
                for j_id in range(self.n_jobs):
                    next_op_id = job_next_op[j_id]
                    if next_op_id < len(self.env.jobs[j_id]['operations']):
                        global_op_idx = self.env.job_op_to_global_op_map.get((j_id, next_op_id))
                        if global_op_idx is not None and scheduled_ops_mask[global_op_idx] == 0:
                            valid_mask[global_op_idx] = 1
                
                if not np.any(valid_mask):
                    break
                
                valid_mask_tensor = torch.from_numpy(valid_mask).float()
                
                # 选择动作
                with torch.no_grad():
                    action, log_prob, _, value = self.policy.get_action_and_value(
                        state, valid_mask_tensor
                    )
                
                op_idx = action.item()
                
                # 执行动作（调度工序）
                op_info = self.env.operations[op_idx]
                job_id, op_id = op_info['job_id'], op_info['op_id']
                precedent_completion_time = op_completion_times.get((job_id, op_id - 1), 0)
                
                best_machine, best_finish_time = -1, float('inf')
                for m_id, proc_time, _, _ in op_info['proc_options']:
                    # 修正 max 用法，确保 machine_release_times[m_id] 为标量
                    m_val = machine_release_times[m_id]
                    if isinstance(m_val, np.ndarray):
                        m_val = float(m_val.item())
                    start_time = m_val if m_val > precedent_completion_time else precedent_completion_time
                    finish_time = start_time + proc_time
                    if finish_time < best_finish_time:
                        best_finish_time = finish_time
                        best_machine = m_id
                
                # 更新环境状态
                machine_release_times[best_machine] = best_finish_time
                op_completion_times[(job_id, op_id)] = best_finish_time
                job_next_op[job_id] += 1
                scheduled_ops_mask[op_idx] = 1
                
                # 计算奖励
                current_makespan = np.max(machine_release_times)
                reward = -(current_makespan - last_makespan)
                last_makespan = current_makespan
                episode_reward += reward
                
                # 判断是否完成
                done = (step == self.n_ops - 1)
                
                # 存储转换
                self.rollout_buffer.push(
                    state, op_idx, log_prob.item(), reward, 
                    value, done, valid_mask_tensor
                )
            
            # 更新最佳makespan
            if last_makespan < best_makespan:
                best_makespan = last_makespan
            
            # 每个episode结束后更新策略
            self._update_policy()
            
            if i_episode > 0 and i_episode % 50 == 0:
                print(f"  PPO | Episode {i_episode}/{self.episodes}, Best Makespan: {best_makespan:.2f}")
        
        # --- 训练结束，使用学习到的策略生成最终解决方案 ---
        print("PPO training finished. Generating final solution...")
        machine_release_times = np.zeros(self.n_machines)
        op_completion_times = {}
        job_next_op = np.zeros(self.n_jobs, dtype=int)
        scheduled_ops_mask = np.zeros(self.n_ops, dtype=int)
        
        final_op_sequence = []
        final_machine_assignment = [-1] * self.n_ops
        
        self.policy.eval()
        with torch.no_grad():
            for _ in range(self.n_ops):
                state = self._get_state(machine_release_times, job_next_op, len(final_op_sequence))
                
                valid_mask = np.zeros(self.n_ops)
                for j_id in range(self.n_jobs):
                    next_op_id = job_next_op[j_id]
                    if next_op_id < len(self.env.jobs[j_id]['operations']):
                        global_op_idx = self.env.job_op_to_global_op_map.get((j_id, next_op_id))
                        if global_op_idx is not None and scheduled_ops_mask[global_op_idx] == 0:
                            valid_mask[global_op_idx] = 1
                
                if not np.any(valid_mask):
                    break
                
                valid_mask_tensor = torch.from_numpy(valid_mask).float()
                
                # 贪婪动作选择（使用最高概率）
                action, _, _, _ = self.policy.get_action_and_value(state, valid_mask_tensor)
                op_idx = action.item()
                
                # 调度工序
                op_info = self.env.operations[op_idx]
                job_id, op_id = op_info['job_id'], op_info['op_id']
                precedent_completion_time = op_completion_times.get((job_id, op_id - 1), 0)
                
                best_machine, best_finish_time = -1, float('inf')
                for m_id, proc_time, _, _ in op_info['proc_options']:
                    # 修正 max 用法，确保 machine_release_times[m_id] 为标量
                    m_val = machine_release_times[m_id]
                    if isinstance(m_val, np.ndarray):
                        m_val = float(m_val.item())
                    start_time = m_val if m_val > precedent_completion_time else precedent_completion_time
                    finish_time = start_time + proc_time
                    if finish_time < best_finish_time:
                        best_finish_time = finish_time
                        best_machine = m_id
                
                machine_release_times[best_machine] = best_finish_time
                op_completion_times[(job_id, op_id)] = best_finish_time
                job_next_op[job_id] += 1
                scheduled_ops_mask[op_idx] = 1
                final_op_sequence.append(op_idx)
                final_machine_assignment[op_idx] = best_machine
        
        # 随机分配运输工具
        final_transport_assignment = [random.randint(0, self.env.num_transporters - 1) 
                                      for _ in range(self.n_jobs)]
        
        final_solution = {
            "op_sequence": final_op_sequence,
            "machine_assignment": final_machine_assignment,
            "transport_assignment": final_transport_assignment
        }
        final_results = self.env.evaluate_solution(final_solution)
        
        print(f"{self.__class__.__name__} finished.")
        return final_solution, final_results