import time
import numpy as np
from collections import defaultdict
import torch
import torch.optim as optim
from collections import defaultdict, deque
from environment import WarehouseEnvironment
from rule_based_agent import RuleBasedDQNAgent, DQNNetwork, ReplayBuffer
from dispatch_heuristic import DispatchHeuristic
from case_generator import FlexibleJobShopScenario
from config import Config
from typing import Dict, List, Tuple
from algorithm_results_saver import save_algorithm_results_csv, generate_instance_id, set_random_seed
import random
import torch.nn as nn
import torch.nn.functional as F

# 训练参数
NUM_EPISODES = 1
BATCH_SIZE = 32
GAMMA = 0.99
LR = 0.0001
TARGET_UPDATE = 100
REPLAY_BUFFER_SIZE = 10000

class DQNAgent:
    """基于Rule-Based Agent状态和动作定义的DQN智能体"""
    
    def __init__(self, config):
        self.config = config
        self.state_dim = 10  # 状态维度
        self.action_dim = 8  # 8种调度规则
        
        # DQN网络
        self.q_net = DQNNetwork(self.state_dim, 128, self.action_dim)
        self.target_q_net = DQNNetwork(self.state_dim, 128, self.action_dim)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=LR)
        self.dispatch_agent = DispatchHeuristic()
        
        # 训练参数
        self.gamma = GAMMA
        self.epsilon_start = 0.3
        self.epsilon_end = 0.01
        self.epsilon_decay = 10000
        self.epsilon = self.epsilon_start
        self.batch_size = BATCH_SIZE
        self.target_update = TARGET_UPDATE
        self.count = 0
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        
        # 训练统计
        self.loss_list = []
        self.episode_rewards = []

    def _get_state(self, state: Dict) -> np.ndarray:
        """将环境状态转换为DQN输入状态 - 基于rule-based agent的实现"""
        jobs = state['available_jobs']
        machines = state['machines']
        
        # 提取关键特征
        avg_due_date = np.mean([j.due_date for j in jobs if hasattr(j, 'due_date')]) if jobs else 0
        proc_times = []
        if jobs:
            for j in jobs:
                for op in j.operations:
                    proc_times.extend(list(op.processing_times.values()))
        avg_proc_time = np.mean(proc_times) if proc_times else 0
        machine_util = np.mean([1 if m.status == 'busy' else 0 for m in machines]) if machines else 0
        
        # 安全获取状态字段，提供默认值
        current_time = state.get('current_time', 0)
        completed_jobs = state.get('completed_jobs', [])
        dispatched_jobs = state.get('dispatched_jobs', [])
        last_schedule_time = state.get('last_schedule_time', 0)
        last_batch_time = state.get('last_batch_time', 0)
        
        state_vec = np.array([
            len(jobs),
            len([j for j in jobs if j.status == 'waiting']),
            avg_due_date,
            avg_proc_time,
            machine_util,
            current_time / self.config.max_time_steps,
            len(completed_jobs),
            len(dispatched_jobs),
            last_schedule_time / self.config.max_time_steps,
            last_batch_time / self.config.max_time_steps
        ], dtype=np.float32)
        
        return state_vec

    def select_action(self, state: Dict):
        """选择调度动作
        返回格式: ({'schedule': {job_id: machine_id}}, rule_idx)
        """
        state_vec = self._get_state(state)
        
        # 更新epsilon值(线性衰减)
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                    np.exp(-1. * self.count / self.epsilon_decay)
        
        # epsilon-贪婪策略选择规则
        if np.random.random() < self.epsilon:
            rule_idx = np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                q_values = self.q_net(torch.FloatTensor(state_vec))
                rule_idx = q_values.argmax().item()
        
        # 保存最后使用的规则索引，以便在update时使用
        self.last_rule_idx = rule_idx
        
        # 应用选中的调度规则
        action = self._apply_rule(rule_idx, state)

        if action is None:
            action = {'wait': True}
        
        return action, rule_idx

    def _apply_rule(self, rule_idx: int, state: Dict):
        """应用指定的调度规则
        返回格式: {'schedule': {job_id: machine_id}}
        """
        jobs = [j for j in state['available_jobs'] if j.status == 'waiting']
        machines = state['machines']
        
        if not jobs or not machines:
            return {}
        
        # 根据规则对作业排序
        if rule_idx == 0:   # EDD
            jobs.sort(key=lambda j: j.due_date)
        elif rule_idx == 1: # SPT
            jobs.sort(key=lambda j: min(j.operations[j.current_operation].processing_times.values()))
        elif rule_idx == 2: # LPT
            jobs.sort(key=lambda j: -max(j.operations[j.current_operation].processing_times.values()))
        elif rule_idx == 3: # CR
            current_time = state.get('t', 0)
            jobs.sort(key=lambda j: (j.due_date - current_time) / 
                     sum(min(op.processing_times.values()) for op in j.operations[j.current_operation:]))
        elif rule_idx == 4: # FCFS
            jobs.sort(key=lambda j: j.job_id)
        elif rule_idx == 5: # MWKR
            jobs.sort(key=lambda j: -sum(min(op.processing_times.values()) 
                     for op in j.operations[j.current_operation:]))
        elif rule_idx == 6: # LWKR
            jobs.sort(key=lambda j: sum(min(op.processing_times.values()) 
                     for op in j.operations[j.current_operation:]))
        
        if rule_idx in (0,1,2,3,4,5,6):
            # 分配作业到最早空闲机器
            schedule = {}
            assigned_machines = set()  # 记录已分配作业的机器ID
        
            for job in jobs:
                op = job.operations[job.current_operation]
                for m_id in op.available_machine_ids:
                    if machines[m_id].status == 'waiting' and m_id not in assigned_machines:
                        schedule[job.job_id] = m_id
                        assigned_machines.add(m_id)  # 标记该机器已分配
                        break  # 跳出当前作业的机器分配循环，继续分配下一个作业
                    
            return {'schedule': schedule}

        elif rule_idx ==7:
            action = self.dispatch_agent.select_action(state)
            return action
        
        return None
            

    def update(self, transition_dict: Dict):
        """更新DQN网络"""
        states = torch.FloatTensor([self._get_state(state) for state in transition_dict['states']])
        actions = torch.LongTensor(transition_dict['actions']).view(-1, 1)
        rewards = torch.FloatTensor(transition_dict['rewards']).view(-1, 1)
        next_states = torch.FloatTensor([self._get_state(state) for state in transition_dict['next_states']])
        dones = torch.FloatTensor(transition_dict['dones']).view(-1, 1)
        
        # 计算目标Q值
        with torch.no_grad():
            max_next_q = self.target_q_net(next_states).max(1)[0].view(-1, 1)
            q_targets = rewards + self.gamma * max_next_q * (1 - dones)
        
        # 计算当前Q值
        q_values = self.q_net(states).gather(1, actions)
        
        # 计算损失并更新
        loss = F.mse_loss(q_values, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1
        
        return loss.item()

    def train_episode(self, env, episode):
        """训练一个episode"""
        state = env.reset()
        total_reward = 0
        episode_steps = 0
        done = False
        
        transition_dict = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': []
        }
        
        while not done:
            # 选择动作
            action, rule_idx = self.select_action(state)
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            # 存储经验
            transition_dict['states'].append(state)
            transition_dict['actions'].append(rule_idx)
            transition_dict['rewards'].append(reward)
            transition_dict['next_states'].append(next_state)
            transition_dict['dones'].append(done)
        
            # 更新状态
            state = next_state
            total_reward += reward
            episode_steps += 1
            
            # 检查终止条件
            if episode_steps >= self.config.max_time_steps:
                done = True
        
        # 更新网络
        if len(transition_dict['states']) > 0:
            loss = self.update(transition_dict)
            self.loss_list.append(loss)
        
        self.episode_rewards.append(total_reward)
        
        # 打印训练信息
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, "
                  f"Epsilon: {self.epsilon:.3f}, "
                  f"Loss: {loss if 'loss' in locals() else 0:.4f}, "
                  f"Steps: {episode_steps}")
        
        return total_reward
           

def run_dqn_experiment(config, case, seed):
    set_random_seed(seed)
    config.seed = seed
    instance_id = generate_instance_id(config)
    env = WarehouseEnvironment(config, case)
    agent = DQNAgent(config)
    
    episode_rewards = []
    start_time = time.time()
    
    for episode in range(NUM_EPISODES):
        reward = agent.train_episode(env, episode)
        episode_rewards.append(reward)
    
    # 计算总训练时间
    total_time = time.time() - start_time
    
    # 构建 stats 字典，包含所有需要的指标
    stats = {
        'episode_rewards': episode_rewards,
        'objective_value': -np.mean(episode_rewards),
        'algorithm_type': 'DQN',
        'total_tardiness': getattr(env, 'total_weighted_tardiness', 0.0),
        'machine_utilization': getattr(env, 'machine_utilization', 0.0),
        'final_makespan': env.t,
        'tardy_penalty': getattr(env, 'tardy_penalty', 0.0),
        'early_reward': getattr(env, 'early_reward', 0.0)
    }
    
    print(f"Training completed in {total_time:.2f} seconds.")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Final makespan: {env.t}")
    print(f"Total tardiness: {getattr(env, 'total_weighted_tardiness', 0.0)}")
    print(f"Tardy penalty: {getattr(env, 'tardy_penalty', 0.0)}")

    # 调用保存函数存储实验结果
    save_algorithm_results_csv(
        algo_name="DQN",
        instance_id=instance_id,
        seed=seed,
        config=config,
        stats=stats,
        env=env,
        total_time=total_time
    )
        
if __name__ == "__main__":
    config = Config()
    config.num_initial_jobs = 20
    config.num_dynamic_jobs = 10
    config.num_machines = 10
    config.num_distributors = 5
    
    config.max_time_steps = 1000
  
   
    case = FlexibleJobShopScenario(config)
    # generate 30 random seeds
    seed=1
    run_dqn_experiment(config, case, seed=seed)
