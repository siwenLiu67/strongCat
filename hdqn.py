import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict, deque
import random

# 环境参数
STATE_DIM = 4
ACTION_DIM = 4
SUBGOAL_DIM = 3

# 训练参数
NUM_EPISODES = 1000
INIT_EPSILON = 1.0
FINAL_EPSILON = 0.1
EPS_ANNEAL_STEPS = 10000
BATCH_SIZE = 32
GAMMA = 0.99
LR = 0.0005

class InternalCritic:
    """内部评判器，生成内在奖励"""
    def __init__(self):
        self.subgoal_thresholds = {
            0: lambda s: s[0] > 0.8,  # 示例子目标判断条件
            1: lambda s: s[1] < -0.5,
            2: lambda s: s[2] > 0.3
        }
    
    def get_reward(self, state, subgoal):
        """返回内在奖励（子目标是否达成）"""
        return 1.0 if self.subgoal_thresholds[subgoal](state) else 0.0

class HierarchicalDQN(nn.Module):
    def __init__(self):
        super().__init__()
        # Meta-Controller网络
        self.meta_controller = nn.Sequential(
            nn.Linear(STATE_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, SUBGOAL_DIM)
        )
        
        # Controller网络（目标条件策略）
        self.controller = nn.Sequential(
            nn.Linear(STATE_DIM + SUBGOAL_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, ACTION_DIM)
        )
    
    def get_meta_q(self, state):
        return self.meta_controller(state)
    
    def get_controller_q(self, state, subgoal):
        return self.controller(torch.cat([state, subgoal], dim=1))

class HierarchicalAgent:
    def __init__(self):
        self.net = HierarchicalDQN()
        self.target_net = HierarchicalDQN()
        self.target_net.load_state_dict(self.net.state_dict())
        
        self.optimizer = optim.Adam(self.net.parameters(), lr=LR)
        self.critic = InternalCritic()
        
        # 经验回放
        self.D1 = deque(maxlen=100000)  # Controller经验
        self.D2 = deque(maxlen=50000)   # Meta-Controller经验
        
        # 探索率管理
        self.epsilon_g = {g: INIT_EPSILON for g in range(SUBGOAL_DIM)}
        self.epsilon_meta = INIT_EPSILON
        self.steps_done = 0
        
        # 子目标跟踪
        self.subgoal_success = defaultdict(lambda: {'attempts': 0, 'successes': 0})

    def select_action(self, state, subgoal):
        """Controller的epsilon-greedy策略"""
        state_tensor = torch.FloatTensor(state)
        subgoal_tensor = torch.zeros(SUBGOAL_DIM)
        subgoal_tensor[subgoal] = 1.0
        
        if random.random() < self.epsilon_g[subgoal]:
            return random.randint(0, ACTION_DIM-1)
        else:
            with torch.no_grad():
                q_values = self.net.get_controller_q(state_tensor.unsqueeze(0), 
                                                     subgoal_tensor.unsqueeze(0))
                return q_values.argmax().item()

    def select_subgoal(self, state):
        """Meta-Controller的epsilon-greedy策略"""
        if random.random() < self.epsilon_meta:
            return random.randint(0, SUBGOAL_DIM-1)
        else:
            with torch.no_grad():
                q_values = self.net.get_meta_q(torch.FloatTensor(state).unsqueeze(0))
                return q_values.argmax().item()

    def update_networks(self):
        # 更新Controller
        if len(self.D1) >= BATCH_SIZE:
            batch = random.sample(self.D1, BATCH_SIZE)
            states, subgoals, actions, rewards, next_states = zip(*batch)
            
            states = torch.FloatTensor(np.array(states))
            subgoals = torch.FloatTensor(np.array(subgoals))
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(np.array(next_states))
            
            current_q = self.net.get_controller_q(states, subgoals).gather(1, actions)
            with torch.no_grad():
                next_q = self.target_net.get_controller_q(next_states, subgoals).max(1)[0]
                target = rewards + GAMMA * next_q
            
            loss = nn.MSELoss()(current_q.squeeze(), target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # 更新Meta-Controller
        if len(self.D2) >= BATCH_SIZE:
            batch = random.sample(self.D2, BATCH_SIZE)
            states, subgoals, F, next_states = zip(*batch)
            
            states = torch.FloatTensor(np.array(states))
            subgoals = torch.LongTensor(subgoals)
            F = torch.FloatTensor(F)
            next_states = torch.FloatTensor(np.array(next_states))
            
            current_q = self.net.get_meta_q(states)[range(BATCH_SIZE), subgoals]
            with torch.no_grad():
                next_q = self.target_net.get_meta_q(next_states).max(1)[0]
                target = F + GAMMA * next_q
            
            loss = nn.MSELoss()(current_q, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # 更新目标网络
        if self.steps_done % 1000 == 0:
            self.target_net.load_state_dict(self.net.state_dict())

    def anneal_epsilon(self, episode):
        # Meta-Controller退火
        self.epsilon_meta = max(FINAL_EPSILON, INIT_EPSILON - 
                               (INIT_EPSILON - FINAL_EPSILON) * episode / NUM_EPISODES)
        
        # Controller自适应退火
        for g in range(SUBGOAL_DIM):
            attempts = self.subgoal_success[g]['attempts']
            successes = self.subgoal_success[g]['successes']
            if attempts > 0:
                success_rate = successes / attempts
                self.epsilon_g[g] = max(FINAL_EPSILON, 1.0 - success_rate)

    def train_episode(self, env):
        state = env.reset()
        total_extrinsic = 0
        episode_steps = 0
        
        while True:
            # Meta-Controller选择子目标
            subgoal = self.select_subgoal(state)
            initial_state = state.copy()
            F = 0  # 累积外在奖励
            goal_achieved = False
            
            # Controller执行循环
            for _ in range(20):  # 子目标最大持续时间
                action = self.select_action(state, subgoal)
                next_state, extrinsic, done, _ = env.step(action)
                intrinsic = self.critic.get_reward(next_state, subgoal)
                
                # 存储Controller经验
                subgoal_onehot = np.zeros(SUBGOAL_DIM)
                subgoal_onehot[subgoal] = 1
                self.D1.append((
                    state, subgoal_onehot, action, intrinsic, next_state
                ))
                
                # 更新统计
                self.subgoal_success[subgoal]['attempts'] += 1
                if intrinsic > 0:
                    self.subgoal_success[subgoal]['successes'] += 1
                    goal_achieved = True
                
                F += extrinsic
                state = next_state
                episode_steps += 1
                self.steps_done += 1
                
                # 更新网络
                self.update_networks()
                
                if done or goal_achieved:
                    break
            
            # 存储Meta-Controller经验
            self.D2.append((
                initial_state, subgoal, F, state
            ))
            
            if done:
                break
        
        # 退火探索率
        self.anneal_epsilon(episode)
        return total_extrinsic

# 训练循环
env = YourEnvironment()  # 需要自定义环境实现
agent = HierarchicalAgent()

for episode in range(NUM_EPISODES):
    reward = agent.train_episode(env)
    print(f"Episode {episode}, Total Reward: {reward:.2f}, "
          f"Meta Epsilon: {agent.epsilon_meta:.3f}, "
          f"Subgoal Success Rates: { {k: v['successes']/(v['attempts']+1e-5) for k, v in agent.subgoal_success.items()} }")
