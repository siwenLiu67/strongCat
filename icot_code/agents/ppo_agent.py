# agents/ppo_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# 检查是否有可用的 CUDA 设备 (GPU)，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    """
    PPO 的 Actor-Critic 网络。
    """
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Actor 网络，输出动作的概率分布
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # 在 actor 网络的最后一层添加一个 hook，以在 Softmax 之前 clamp 输入值
        self.actor[-2].register_forward_hook(lambda module, input, output: output.clamp(-10, 10))
        
        # Critic 网络，评估状态的价值
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 在 critic 网络的最后一层添加一个 hook，以在 Softmax 之前 clamp 输入值
        self.critic[-2].register_forward_hook(lambda module, input, output: output.clamp(-10, 10))

    def forward(self, state):
        # 健壮性检查：检查并处理输入状态中的 NaN 或 Inf
        if torch.isnan(state).any() or torch.isinf(state).any():
            print(f"  [ActorCritic Warning] Detected NaN or Inf in input state. Replacing with 0.")
            state = torch.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
            
        x = self.actor[0](state)
        x = self.actor[1](x)
        x = self.actor[2](x)
        x = self.actor[3](x)
        x = self.actor[4](x)
        action_probs = self.actor[5](x)
        
        # 调试：检查 actor 网络的输出
        if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
            print(f"  [ActorCritic Debug] Detected NaN or Inf in action_probs: {action_probs}")
            
        state_value = self.critic(state)
        return action_probs, state_value

class PPOAgent:
    def __init__(self, state_dim, action_dim, drl_hyperparams):
        self.gamma = drl_hyperparams['gamma']
        self.lr = drl_hyperparams['learning_rate']
        
        # 将网络移动到指定设备
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        self.memory = []  # 用于存储经验的缓冲区
        self.eps_clip = 0.2  # PPO裁剪参数

    def select_action(self, state):
        """根据当前策略选择一个动作。"""
        # 将状态张量移动到设备
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_probs, _ = self.policy_old(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.item(), action_logprob

    def store_transition(self, state, action, log_prob, reward, done):
        """将一个经验转换存储到内存中。"""
        self.memory.append((state, action, log_prob, reward, done))

    def update(self):
        """更新策略网络。"""
        # 如果内存中的经验太少，则跳过更新，以避免 NaN 问题
        if len(self.memory) <= 1:
            self.memory = [] # 清空内存
            print("  [PPO Debug] Skipping update due to insufficient experience.")
            return

        # 蒙特卡洛估计奖励:
        rewards = []
        discounted_reward = 0
        for _, _, _, reward, done in reversed(self.memory):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # 标准化奖励并移动到设备
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        
        # 标准化奖励并移动到设备
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards_std = rewards.std()

        # 健壮性检查：仅在标准差大于一个很小的值时才进行标准化
        if rewards_std > 1e-6:
            rewards = (rewards - rewards.mean()) / rewards_std
        else:
            # 如果标准差为零或非常小，则不进行标准化（或仅进行中心化）
            rewards = rewards - rewards.mean()

        # 调试：检查标准化后的奖励
        if torch.isnan(rewards).any() or torch.isinf(rewards).any():
            print(f"  [PPO Debug] Detected NaN or Inf in normalized rewards: {rewards}")
            # 可以选择在这里强制终止训练或跳过更新
            
        # 将列表转换为张量并移动到设备
        old_states = torch.tensor(np.array([t[0] for t in self.memory]), dtype=torch.float32).to(device)
        old_actions = torch.tensor([t[1] for t in self.memory], dtype=torch.int64).to(device)
        old_logprobs = torch.tensor([t[2] for t in self.memory], dtype=torch.float32).to(device)

        # 对策略进行 K 个周期的优化
        for _ in range(5): # K_epochs
            # 调试：检查 old_states 中是否存在 NaN 或 Inf
            if torch.isnan(old_states).any() or torch.isinf(old_states).any():
                print(f"  [PPO Debug] Detected NaN or Inf in old_states before policy evaluation: {old_states}")
                # 可以选择在这里强制终止训练或跳过更新
                
            # 评估旧的动作和价值
            action_probs, state_values = self.policy(old_states)
            
            # 调试：检查 action_probs 中是否存在 NaN 或 Inf
            if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
                print(f"  [PPO Debug] Detected NaN or Inf in action_probs: {action_probs}")
                # 可以选择在这里强制终止训练或跳过更新
                
            dist = Categorical(action_probs)
            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            
            # 计算比率 (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # 计算替代损失 (Surrogate Loss)
            advantages = rewards - state_values.squeeze()
            
            # 调试：检查优势值
            if torch.isnan(advantages).any() or torch.isinf(advantages).any():
                print(f"  [PPO Debug] Detected NaN or Inf in advantages: {advantages}")
                # 可以选择在这里强制终止训练或跳过更新
                
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values.squeeze(), rewards) - 0.01 * dist_entropy
            
            # 调试：检查损失
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"  [PPO Debug] Detected NaN or Inf in loss: {loss}")
                # 可以选择在这里强制终止训练或跳过更新
                
            # 执行梯度步骤
            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5) # 添加梯度裁剪
            self.optimizer.step()

        # 将新权重复制到旧策略中
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.memory = []


def DRL_Scheduling_Agent(env, agent, max_episodes):
    """
    DRL代理的主训练循环。
    """
    best_c_max = float('inf')
    best_schedule = None
    rewards = []

    for episode in range(max_episodes):
        state = env.reset()
        done = False
        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, log_prob, reward, done)
            state = next_state
        
        rewards.append(reward)

        agent.update()
        
        if env.C_max < best_c_max:
            best_c_max = env.C_max
            best_schedule = env.schedule

        # 在每个轮次后打印日志，以显示详细的训练进度
        print(f"    - 轮次 {episode+1}/{max_episodes}, C_max: {env.C_max:.2f}, 当前最佳 C_max: {best_c_max:.2f}")

    return best_schedule, best_c_max, rewards

if __name__ == '__main__':
    import sys
    import os
    # 将项目根目录添加到 sys.path，以便正确导入 config 和 models 模块
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from models.production_env import FJSPEnv
    from data_loader import load_production_data, load_drl_hyperparameters, load_transportation_data

    prod_data = load_production_data()
    drl_params = load_drl_hyperparameters()
    trans_data = load_transportation_data() # 加载运输数据以获取订单信息
    
    env = FJSPEnv(prod_data, trans_data['orders'], T_internal=50)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = PPOAgent(state_dim, action_dim, drl_params)
    
    print("--- 开始 DRL 代理训练 ---")
    schedule, c_max = DRL_Scheduling_Agent(env, agent, drl_params['max_episodes'])
    
    print("\n--- 训练完成 ---")
    print(f"最优生产调度 (C_max = {c_max}):")
    # for entry in schedule:
    #     print(entry)
