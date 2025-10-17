"""
Double Dueling DQN (D3QN)
结合了 Double DQN 和 Dueling DQN 的优势
"""
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import math
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


class DuelingDQN_Model(nn.Module):
    """
    Dueling DQN 网络架构
    将Q值分解为状态价值V(s)和优势函数A(s,a)
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
    """
    def __init__(self, n_obs, n_actions, hidden_dim=128):
        super(DuelingDQN_Model, self).__init__()
        
        # 共享特征提取层
        self.feature_layer = nn.Sequential(
            nn.Linear(n_obs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 状态价值流 (Value Stream)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 优势函数流 (Advantage Stream)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions)
        )
        
    def forward(self, x):
        """前向传播"""
        features = self.feature_layer(x)
        
        # 计算状态价值
        value = self.value_stream(features)
        
        # 计算优势函数
        advantages = self.advantage_stream(features)
        
        # 组合成Q值: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        # 减去平均值是为了可识别性（identifiability）
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values
    
    def get_value_and_advantage(self, x):
        """分别获取价值和优势（用于分析）"""
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return value, advantages


class PrioritizedReplayMemory:
    """
    优先经验回放（可选增强）
    根据TD误差优先采样重要的经验
    """
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha  # 优先级指数
        self.memory = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
    def push(self, *args):
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        if self.size < self.capacity:
            self.memory.append(Transition(*args))
            self.size += 1
        else:
            self.memory[self.position] = Transition(*args)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        """根据优先级采样"""
        if self.size < batch_size:
            return None, None, None
        
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        samples = [self.memory[idx] for idx in indices]
        
        # 计算重要性采样权重
        total = self.size
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, indices, torch.FloatTensor(weights)
    
    def update_priorities(self, indices, priorities):
        """更新经验的优先级"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-5
    
    def __len__(self):
        return self.size


class D3QN_Agent(BaseAlgorithm):
    """
    Double Dueling DQN (D3QN) 智能体
    结合了以下技术:
    1. Double DQN: 使用目标网络选择动作，policy网络评估Q值，减少过高估计
    2. Dueling DQN: 分离状态价值和优势函数，提高学习效率
    3. 优先经验回放: 优先学习重要的经验
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
        self.eps_decay = self.config.get('eps_decay', 5000)  # 原1000，改为5000，探索期更长
        self.lr = self.config.get('lr', 1e-4)
        self.target_update_freq = self.config.get('target_update_freq', 5)
        self.use_prioritized_replay = self.config.get('use_prioritized_replay', True)
        memory_size = self.config.get('memory_size', 10000)
        
        
        # 状态维度：机器释放时间 + 每个作业的下一道工序索引 + 已完成工序比例 + 新增3个特征
        self.state_dim = self.n_machines + self.n_jobs + 1 + self.n_machines + self.n_jobs + self.n_machines
        
        # 初始化Dueling DQN网络
        self.policy_net = DuelingDQN_Model(self.state_dim, self.n_ops)
        self.target_net = DuelingDQN_Model(self.state_dim, self.n_ops)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 目标网络设为评估模式
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        
        # 选择经验回放方式
        if self.use_prioritized_replay:
            self.memory = PrioritizedReplayMemory(memory_size)
            self.beta_start = 0.4
            self.beta_end = 1.0
            self.beta_increment = (self.beta_end - self.beta_start) / self.episodes
            self.beta = self.beta_start
        else:
            self.memory = ReplayMemory(memory_size)
        
        self.steps_done = 0

        # 新增n_step参数
        self.n_step = self.config.get('n_step', 5)
        self.n_step_buffer = []

    def select_action(self, state, valid_mask, machine_release_times=None, op_completion_times=None):
        """结合 Q 值和调度启发式分数（finish_time）加权选择动作"""
        eps = self.eps_end + (self.eps_start - self.eps_end) * \
              math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if random.random() > eps:
            # 利用：Q值与启发式分数加权
            with torch.no_grad():
                q_values = self.policy_net(state)[0].cpu().numpy()
                q_values[valid_mask == 0] = -float('inf')
                valid_actions = np.where(valid_mask == 1)[0]
                finish_times = []
                for op_idx in valid_actions:
                    op_info = self.env.operations[op_idx]
                    job_id, op_id = op_info['job_id'], op_info['op_id']
                    precedent_completion_time = 0
                    if op_completion_times is not None:
                        precedent_completion_time = op_completion_times.get((job_id, op_id - 1), 0)
                    best_finish_time = float('inf')
                    for m_id, proc_time, _, _ in op_info['proc_options']:
                        m_val = machine_release_times[m_id] if machine_release_times is not None else 0
                        start_time = max(m_val, precedent_completion_time)
                        finish_time = start_time + proc_time
                        if finish_time < best_finish_time:
                            best_finish_time = finish_time
                    finish_times.append(best_finish_time)
                q_valid = q_values[valid_actions]
                ft_valid = np.array(finish_times)
                norm_q = (q_valid - np.min(q_valid)) / (np.ptp(q_valid) + 1e-9) if np.ptp(q_valid) > 0 else np.zeros_like(q_valid)
                norm_ft = (ft_valid - np.min(ft_valid)) / (np.ptp(ft_valid) + 1e-9) if np.ptp(ft_valid) > 0 else np.zeros_like(ft_valid)
                
                alpha = 0.2
                scores = alpha * norm_q + (1 - alpha) * (1 - norm_ft)
                # Top-K softmax采样
                K = min(3, len(scores))
                topk_idx = np.argpartition(-scores, K-1)[:K]
                topk_scores = scores[topk_idx]
                exp_scores = np.exp(topk_scores - np.max(topk_scores))
                probs = exp_scores / (np.sum(exp_scores) + 1e-9)
                chosen = np.random.choice(K, p=probs)
                best_action = valid_actions[topk_idx[chosen]]
                return torch.tensor([[best_action]], dtype=torch.long)
        else:
            valid_actions = np.where(valid_mask == 1)[0]
            return torch.tensor([[random.choice(valid_actions)]], dtype=torch.long)

    def _optimize_model(self):
        """优化模型 - 实现Double DQN + Dueling架构"""
        if isinstance(self.memory, PrioritizedReplayMemory):
            if len(self.memory) < self.batch_size:
                return
            samples, indices, weights = self.memory.sample(self.batch_size, self.beta)
            if samples is None or weights is None:
                return
            transitions = samples
            batch = Transition(*zip(*transitions))
            weights = weights.unsqueeze(1)
        else:
            if len(self.memory) < self.batch_size:
                return
            transitions = self.memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))
            weights = torch.ones(self.batch_size, 1)
        
        # 准备批次数据
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), 
            dtype=torch.bool
        )
        
        # 修正经验回放，过滤掉 NoneType 状态
        batch_state_list = [s for s in batch.state if s is not None]
        if len(batch_state_list) < self.batch_size:
            return
        state_batch = torch.cat(batch_state_list)
        
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        # 使用policy网络计算当前Q值
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Double DQN: 使用policy网络选择动作，target网络评估Q值
        next_state_values = torch.zeros(self.batch_size)
        
        if non_final_mask.any():
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
            
            with torch.no_grad():
                # 使用policy网络选择最佳动作
                next_actions = self.policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
                
                # 使用target网络评估选定动作的Q值
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_actions).squeeze()
        
        # 计算期望Q值
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
        # 计算TD误差（用于优先经验回放）
        td_errors = torch.abs(state_action_values - expected_state_action_values.unsqueeze(1))
        
        # 计算加权Huber损失
        loss = (weights * nn.SmoothL1Loss(reduction='none')(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )).mean()
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        # 仅 PrioritizedReplayMemory 类型才调用 update_priorities
        if isinstance(self.memory, PrioritizedReplayMemory):
            priorities = td_errors.detach().cpu().numpy().flatten()
            self.memory.update_priorities(indices, priorities)

    def _get_state(self, machine_release_times, job_next_op, scheduled_ops_count):
        """构建状态表示，增加3个新特征：机器负载、作业剩余工序数、机器累计加工时间"""
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
        # 新增特征1：每台机器当前已分配工序数（近似：release_time>0即有分配）
        machine_loads = (m_times > 0).astype(np.float32)
        machine_loads_norm = machine_loads / (np.max(machine_loads) + 1e-9) if np.max(machine_loads) > 0 else machine_loads
        # 新增特征2：每个作业剩余工序数
        job_remaining_ops = np.array([
            max_ops_per_job - j_ops[j] for j in range(self.n_jobs)
        ])
        job_remaining_ops_norm = job_remaining_ops / (max_ops_per_job + 1e-9) if max_ops_per_job > 0 else job_remaining_ops
        # 新增特征3：每台机器累计加工时间
        machine_total_proc_time = np.copy(machine_release_times)
        machine_total_proc_time_norm = machine_total_proc_time / (np.max(machine_total_proc_time) + 1e-9) if np.max(machine_total_proc_time) > 0 else machine_total_proc_time


        # 拼接所有特征
        state_np = np.concatenate([
            m_times_norm, j_ops_norm, progress,
            machine_loads_norm, job_remaining_ops_norm, machine_total_proc_time_norm
        ]).astype(np.float32)
        return torch.from_numpy(state_np).unsqueeze(0)


   
    def solve(self):
        """
        训练 D3QN 智能体并返回最佳解决方案
        """
        print(f"Running {self.__class__.__name__} (Double Dueling DQN) with training for {self.episodes} episodes...")
        
        best_makespan = float('inf')
        best_solution = None
        best_results = None
        # --- 训练循环 ---
        for i_episode in range(self.episodes):
            # 更新beta（用于优先经验回放）
            if self.use_prioritized_replay:
                self.beta = min(self.beta_end, self.beta + self.beta_increment)
            
            # --- 重置环境 ---
            machine_release_times = np.zeros(self.n_machines)
            op_completion_times = {}
            job_next_op = np.zeros(self.n_jobs, dtype=int)
            scheduled_ops_mask = np.zeros(self.n_ops, dtype=int)
            
            state = self._get_state(machine_release_times, job_next_op, 0)
            last_makespan = 0

            for step in range(self.n_ops):
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

                # 选择并执行动作
                action_tensor = self.select_action(state, valid_mask, machine_release_times, op_completion_times)
                op_idx = action_tensor.item()
                
                # 找到最佳机器（最早完成时间启发式）
                op_info = self.env.operations[op_idx]
                job_id, op_id = op_info['job_id'], op_info['op_id']
                precedent_completion_time = op_completion_times.get((job_id, op_id - 1), 0)

                best_machine, best_finish_time = -1, float('inf')
                for m_id, proc_time, _, _ in op_info['proc_options']:
                    # 修正 max 用法，确保取标量且类型安全
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
        
               
                # 计算奖励（每步都用当前 makespan，鼓励全局更小）
                current_makespan = np.max(machine_release_times)
                reward = -(current_makespan - last_makespan)  # 负的 makespan 作为奖励
                last_makespan = current_makespan
                
                # 观察新状态并存储转换
                done = (step == self.n_ops - 1)
                next_state = None if done else self._get_state(
                    machine_release_times, job_next_op, step + 1
                )
                
                # n-step buffer追加
                self.n_step_buffer.append((state, action_tensor, reward, next_state))
                if len(self.n_step_buffer) >= self.n_step:
                    R = sum([self.n_step_buffer[i][2] * (self.gamma ** i) for i in range(self.n_step)])
                    s, a, _, _ = self.n_step_buffer[0]
                    _, _, _, ns = self.n_step_buffer[-1]
                    self.memory.push(s, a, ns, torch.tensor([R], dtype=torch.float32))
                    self.n_step_buffer.pop(0)
                state = next_state

                # 优化模型
                self._optimize_model()

            # episode结束后清空n步buffer，补充剩余经验
            while len(self.n_step_buffer) > 0:
                i = len(self.n_step_buffer)
                R = sum([self.n_step_buffer[j][2] * (self.gamma ** j) for j in range(i)])
                s, a, _, _ = self.n_step_buffer[0]
                _, _, _, ns = self.n_step_buffer[-1]
                self.memory.push(s, a, ns, torch.tensor([R], dtype=torch.float32))
                self.n_step_buffer.pop(0)

            # 定期更新目标网络
            tau = 0.005
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

            
            if i_episode > 0 and i_episode % 50 == 0:
                print(f"  D3QN | Episode {i_episode}/{self.episodes}, Epsilon: {self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay):.3f}")

            # episode结束后评估当前策略
            current_solution = {
                "op_sequence": [],
                "machine_assignment": [-1] * self.n_ops,
                "transport_assignment": [random.randint(0, self.env.num_transporters - 1) for _ in range(self.n_jobs)]
            }
            machine_release_times_eval = np.zeros(self.n_machines)
            op_completion_times_eval = {}
            job_next_op_eval = np.zeros(self.n_jobs, dtype=int)
            scheduled_ops_mask_eval = np.zeros(self.n_ops, dtype=int)
            for step_eval in range(self.n_ops):
                valid_mask_eval = np.zeros(self.n_ops)
                for j_id in range(self.n_jobs):
                    next_op_id = job_next_op_eval[j_id]
                    if next_op_id < len(self.env.jobs[j_id]['operations']):
                        global_op_idx = self.env.job_op_to_global_op_map.get((j_id, next_op_id))
                        if global_op_idx is not None and scheduled_ops_mask_eval[global_op_idx] == 0:
                            valid_mask_eval[global_op_idx] = 1
                if not np.any(valid_mask_eval):
                    break
                action_tensor_eval = self.select_action(self._get_state(machine_release_times_eval, job_next_op_eval, step_eval), valid_mask_eval, machine_release_times_eval, op_completion_times_eval)
                op_idx_eval = action_tensor_eval.item()
                op_info_eval = self.env.operations[op_idx_eval]
                job_id_eval, op_id_eval = op_info_eval['job_id'], op_info_eval['op_id']
                precedent_completion_time_eval = op_completion_times_eval.get((job_id_eval, op_id_eval - 1), 0)
                best_machine_eval, best_finish_time_eval = -1, float('inf')
                for m_id, proc_time, _, _ in op_info_eval['proc_options']:
                    m_val = machine_release_times_eval[m_id]
                    if isinstance(m_val, np.ndarray):
                        m_val = float(m_val.item())
                    start_time = max(m_val, precedent_completion_time_eval)
                    finish_time = start_time + proc_time
                    if finish_time < best_finish_time_eval:
                        best_finish_time_eval = finish_time
                        best_machine_eval = m_id
                machine_release_times_eval[best_machine_eval] = best_finish_time_eval
                op_completion_times_eval[(job_id_eval, op_id_eval)] = best_finish_time_eval
                job_next_op_eval[job_id_eval] += 1
                scheduled_ops_mask_eval[op_idx_eval] = 1
                current_solution["op_sequence"].append(op_idx_eval)
                current_solution["machine_assignment"][op_idx_eval] = best_machine_eval
            # 评估当前解
            current_results = self.env.evaluate_solution(current_solution)
            current_makespan = current_results.get('makespan', float('inf'))
            # 保存历史最优
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_solution = current_solution.copy()
                best_results = current_results.copy()
            # 在 episode 结束时追加全局负 makespan reward
            if self.memory:
                self.memory.push(state, action_tensor, None, torch.tensor([-current_makespan], dtype=torch.float32))

        # --- 训练结束，使用历史最优解生成最终解决方案 ---
        print("D3QN training finished. Generating final solution...")
        if best_solution is not None and best_results is not None:
            print('D3QN op_sequence:', best_solution["op_sequence"])
            print('D3QN machine_assignment:', best_solution["machine_assignment"])
            print('D3QN transport_assignment:', best_solution["transport_assignment"])
            print('D3QN makespan:', best_results.get('makespan'))
            print('D3QN objective_value:', best_results.get('objective_value'))
            print('D3QN job_final_completion_times:', best_results.get('job_final_completion_times', 'N/A'))
            print('D3QN machine_release_times:', machine_release_times_eval)
            print('D3QN op_completion_times:', op_completion_times_eval)
            return best_solution, best_results
        else:
            # 若无历史最优，则重新生成一次解
            machine_release_times = np.zeros(self.n_machines)
            op_completion_times = {}
            job_next_op = np.zeros(self.n_jobs, dtype=int)
            scheduled_ops_mask = np.zeros(self.n_ops, dtype=int)
            final_op_sequence = []
            final_machine_assignment = [-1] * self.n_ops
            self.policy_net.eval()
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
                    q_values = self.policy_net(state)
                    q_values[0, valid_mask == 0] = -float('inf')
                    op_idx = q_values.max(1)[1].item()
                    op_info = self.env.operations[op_idx]
                    job_id, op_id = op_info['job_id'], op_info['op_id']
                    precedent_completion_time = op_completion_times.get((job_id, op_id - 1), 0)
                    best_machine, best_finish_time = -1, float('inf')
                    for m_id, proc_time, _, _ in op_info['proc_options']:
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
            final_transport_assignment = [random.randint(0, self.env.num_transporters - 1) for _ in range(self.n_jobs)]
            final_solution = {
                "op_sequence": final_op_sequence,
                "machine_assignment": final_machine_assignment,
                "transport_assignment": final_transport_assignment
            }
            final_results = self.env.evaluate_solution(final_solution)
            print('D3QN op_sequence:', final_op_sequence)
            print('D3QN machine_assignment:', final_machine_assignment)
            print('D3QN transport_assignment:', final_transport_assignment)
            print('D3QN makespan:', final_results.get('makespan'))
            print('D3QN objective_value:', final_results.get('objective_value'))
            print('D3QN job_final_completion_times:', final_results.get('job_final_completion_times', 'N/A'))
            print('D3QN machine_release_times:', machine_release_times)
            print('D3QN op_completion_times:', op_completion_times)
            return final_solution, final_results