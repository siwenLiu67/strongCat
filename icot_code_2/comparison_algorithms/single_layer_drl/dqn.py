"""
深度Q网络 (Deep Q-Network)
"""
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import math
import copy
from ..base_algorithm import BaseAlgorithm

# 定义经验回放的转换单元
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    """经验回放缓冲区"""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN_Model(nn.Module):
    """DQN 的神经网络模型"""
    def __init__(self, n_obs, n_actions):
        super(DQN_Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_obs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.layers(x)

class DQN_Agent(BaseAlgorithm):
    """
    使用 DQN 求解 FJSP 的智能体。
    """
    def __init__(self, env, config):
        """
        :param env: FjspEnv, a scheduling environment instance.
        :param config: dict, a dictionary containing algorithm-specific hyperparameters.
        """
        super().__init__(env, config)
        # 从配置中加载超参数
        self.episodes = self.config.get('episodes', 500)
        self.batch_size = self.config.get('batch_size', 128)
        self.gamma = self.config.get('gamma', 0.99)
        self.eps_start = self.config.get('eps_start', 0.9)
        self.eps_end = self.config.get('eps_end', 0.05)
        self.eps_decay = self.config.get('eps_decay', 1000)
        self.lr = self.config.get('lr', 1e-4)
        memory_size = self.config.get('memory_size', 10000)
        
        # 状态维度：机器释放时间 + 每个作业的下一道工序索引 + 已完成工序比例
        self.state_dim = self.n_machines + self.n_jobs + 1
        
        self.policy_net = DQN_Model(self.state_dim, self.n_ops)
        self.target_net = DQN_Model(self.state_dim, self.n_ops)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(memory_size)
        self.steps_done = 0

    def select_action(self, state, valid_mask):
        """使用 epsilon-greedy策略选择动作"""
        eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if random.random() > eps:
            with torch.no_grad():
                q_values = self.policy_net(state)
                q_values[0, valid_mask == 0] = -float('inf') # 屏蔽无效动作
                return q_values.max(1)[1].view(1, 1)
        else:
            # 选择一个随机的有效动作
            valid_actions = np.where(valid_mask == 1)[0]
            return torch.tensor([[random.choice(valid_actions)]], dtype=torch.long)

    def _optimize_model(self):
        """优化模型"""
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        next_state_values = torch.zeros(self.batch_size)
        if non_final_mask.any():
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def _get_state(self, machine_release_times, job_next_op, scheduled_ops_count):
        # Normalize inputs to create a state representation
        m_times = np.copy(machine_release_times)
        j_ops = np.copy(job_next_op)
        
        # Normalization to avoid large values
        m_times_norm = m_times / (np.mean(m_times) + 1e-9) if np.mean(m_times) > 0 else m_times
        
        # Find max ops for any job for normalization
        max_ops_per_job = 0
        for job_spec in self.env.jobs:
            if len(job_spec['operations']) > max_ops_per_job:
                max_ops_per_job = len(job_spec['operations'])
        
        j_ops_norm = j_ops / (max_ops_per_job + 1e-9) if max_ops_per_job > 0 else j_ops
        
        progress = np.array([scheduled_ops_count / self.n_ops])
        
        state_np = np.concatenate([m_times_norm, j_ops_norm, progress]).astype(np.float32)
        return torch.from_numpy(state_np).unsqueeze(0)

    def solve(self):
        """
        Trains the DQN agent by interacting with a simulated environment
        and returns the best solution found after training.
        """
        print(f"Running {self.__class__.__name__} with training for {self.episodes} episodes...")

        # --- Training Loop ---
        for i_episode in range(self.episodes):
            # --- Reset environment for a new episode ---
            machine_release_times = np.zeros(self.n_machines)
            op_completion_times = {}  # (job_id, op_id) -> time
            job_next_op = np.zeros(self.n_jobs, dtype=int)
            scheduled_ops_mask = np.zeros(self.n_ops, dtype=int)
            
            state = self._get_state(machine_release_times, job_next_op, 0)
            last_makespan = 0

            for step in range(self.n_ops):
                # Determine valid actions (next operation for each job that hasn't been scheduled)
                valid_mask = np.zeros(self.n_ops)
                for j_id in range(self.n_jobs):
                    next_op_id = job_next_op[j_id]
                    if next_op_id < len(self.env.jobs[j_id]['operations']):
                        global_op_idx = self.env.job_op_to_global_op_map.get((j_id, next_op_id))
                        if global_op_idx is not None and scheduled_ops_mask[global_op_idx] == 0:
                            valid_mask[global_op_idx] = 1
                
                if not np.any(valid_mask): break

                # Select and perform action
                action_tensor = self.select_action(state, valid_mask)
                op_idx = action_tensor.item()
                
                # Find best machine for the chosen op (earliest finish time heuristic)
                op_info = self.env.operations[op_idx]
                job_id, op_id = op_info['job_id'], op_info['op_id']
                precedent_completion_time = op_completion_times.get((job_id, op_id - 1), 0)

                best_machine, best_finish_time = -1, float('inf')
                for m_id, proc_time, _, _ in op_info['proc_options']:
                    start_time = max(machine_release_times[m_id], precedent_completion_time)
                    finish_time = start_time + proc_time
                    if finish_time < best_finish_time:
                        best_finish_time = finish_time
                        best_machine = m_id
                
                # Update environment state
                machine_release_times[best_machine] = best_finish_time
                op_completion_times[(job_id, op_id)] = best_finish_time
                job_next_op[job_id] += 1
                scheduled_ops_mask[op_idx] = 1
                
                # Calculate reward (negative change in makespan)
                current_makespan = np.max(machine_release_times)
                reward = -(current_makespan - last_makespan)
                last_makespan = current_makespan
                
                # Observe new state and store transition
                done = (step == self.n_ops - 1)
                next_state = None if done else self._get_state(machine_release_times, job_next_op, step + 1)
                
                self.memory.push(state, action_tensor, next_state, torch.tensor([reward], dtype=torch.float32))
                state = next_state

                self._optimize_model()

            # Update target network periodically
            if i_episode > 0 and i_episode % 10 == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            if i_episode > 0 and i_episode % 50 == 0:
                print(f"  DQN | Episode {i_episode}/{self.episodes}...")

        # --- After training, generate final solution using the learned policy greedily ---
        print("DQN training finished. Generating final solution...")
        machine_release_times = np.zeros(self.n_machines)
        op_completion_times = {}
        job_next_op = np.zeros(self.n_jobs, dtype=int)
        scheduled_ops_mask = np.zeros(self.n_ops, dtype=int)
        
        final_op_sequence = []
        final_machine_assignment = [-1] * self.n_ops
        
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
                
                if not np.any(valid_mask): break

                # Greedy action selection
                q_values = self.policy_net(state)
                q_values[0, valid_mask == 0] = -float('inf')
                op_idx = q_values.max(1)[1].item()

                # Schedule op_idx
                op_info = self.env.operations[op_idx]
                job_id, op_id = op_info['job_id'], op_info['op_id']
                precedent_completion_time = op_completion_times.get((job_id, op_id - 1), 0)
                
                best_machine, best_finish_time = -1, float('inf')
                for m_id, proc_time, _, _ in op_info['proc_options']:
                    start_time = max(machine_release_times[m_id], precedent_completion_time)
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

        final_transport_assignment = [random.randint(0, self.env.num_transporters - 1) for _ in range(self.n_jobs)]
        final_solution = {
            "op_sequence": final_op_sequence,
            "machine_assignment": final_machine_assignment,
            "transport_assignment": final_transport_assignment
        }
        final_results = self.env.evaluate_solution(final_solution)

        print(f"{self.__class__.__name__} finished.")
        return final_solution, final_results
