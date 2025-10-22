import time
import numpy as np
from collections import defaultdict, deque
import torch
import torch.nn as nn
import torch.optim as optim
import random
from typing import Dict, List, Tuple

from environment import WarehouseEnvironment
from rule_based_agent import RuleBasedDQNAgent 
from dispatch_heuristic import DispatchHeuristic
from case_generator import FlexibleJobShopScenario
from config import Config
from algorithm_results_saver import save_algorithm_results_csv, generate_instance_id, set_random_seed

# 训练参数
NUM_EPISODES = 100
INIT_EPSILON = 0.9
FINAL_EPSILON = 0.01
EPS_ANNEAL_STEPS = 20000
BATCH_SIZE = 128
GAMMA = 0.95
LR = 0.0001

# 网络参数
STATE_DIM = 10  # 状态特征维度
SUBGOAL_DIM = 3  # 子目标维度
ACTION_DIM = 10  # 统一动作空间：8个调度规则 + 1个配送启发式 + 1个等待动作


class InternalCritic:
    """内部评判器，生成内在奖励"""
    def __init__(self, config):
        self.config = config
        self.subgoal_thresholds = {
            0: self._evaluate_production_goal,    # 生产优化子目标
            1: self._evaluate_delivery_goal,      # 配送效率子目标
            2: self._evaluate_balancing_goal      # 系统平衡子目标
        }
    
    def get_reward(self, state, subgoal_idx):
        """返回基于业务逻辑的内在奖励"""
        return self.subgoal_thresholds[subgoal_idx](state)

    def _evaluate_production_goal(self, state):
        """评估生产优化子目标达成度"""
        machine_utilization = state['machine_utilization']
        completed_jobs = len(state['completed_jobs'])
        active_jobs = len(state['available_jobs'])
        
        utilization_score = min(1.0, machine_utilization / 0.8)
        completion_rate = completed_jobs / max(1, completed_jobs + active_jobs)
        
        production_score = (utilization_score * 0.6 + completion_rate * 0.4)
        return production_score * 0.5
    
    def _evaluate_delivery_goal(self, state):
        """评估配送效率子目标达成度"""
        completed_jobs = state['completed_jobs']
        dispatched_jobs = state['dispatched_jobs']
        current_time = state['current_time']
        
        if not completed_jobs:
            return 0.0
        
        total_tardiness = sum(max(0, current_time - j.due_date) for j in completed_jobs)
        avg_tardiness = total_tardiness / len(completed_jobs)
        
        delivery_efficiency = min(1.0, len(dispatched_jobs) / len(completed_jobs))  
        tardiness_penalty = max(0, 1 - avg_tardiness / 100.0)
        
        delivery_score = (delivery_efficiency * 0.7 + tardiness_penalty * 0.3)
        return delivery_score * 0.5
    
    def _evaluate_balancing_goal(self, state):
        """评估系统平衡子目标达成度"""
        machines = state.get('machines', [])
        jobs = state.get('available_jobs', [])
        
        if not machines:
            return 0.0
        
        machine_loads = [m.remaining_time for m in machines]
        load_std = np.std(machine_loads) if machine_loads else 0
        load_balance = max(0, 1 - load_std / self.config.max_processing_time)
        
        if jobs:
            wait_times = [getattr(j, 'waiting_time', 0) for j in jobs]
            wait_std = np.std(wait_times)
            wait_balance = max(0, 1 - wait_std / 50.0)
        else:
            wait_balance = 1.0
        
        system_pressure = len(jobs) / max(1, len(machines) * 3)
        pressure_score = max(0, 1 - system_pressure)
        
        balancing_score = (load_balance * 0.4 + wait_balance * 0.3 + pressure_score * 0.3)
        return balancing_score * 0.3


class HierarchicalDQN(nn.Module):
    """整合的Hierarchical DQN网络"""
    
    def __init__(self):
        super().__init__()
        
        # Meta-Controller网络 - 选择子目标
        self.meta_controller = nn.Sequential(
            nn.Linear(STATE_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, SUBGOAL_DIM)
        )
        
        # Unified Controller网络 - 根据子目标选择具体动作
        self.controller = nn.Sequential(
            nn.Linear(STATE_DIM + SUBGOAL_DIM, 128),  # 状态 + 子目标
            nn.ReLU(),
            nn.Dropout(0.2), # 防止过拟合
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, ACTION_DIM)
        )
    
    def forward(self, state):
        """前向传播 - 返回Meta-Controller的Q值"""
        return self.meta_controller(state)
    
    def get_meta_q(self, state):
        """获取Meta-Controller的Q值"""
        return self.meta_controller(state)
    
    def get_controller_q(self, state, subgoal):
        """获取Controller的Q值 - 状态和子目标连接"""
        combined = torch.cat([state, subgoal], dim=1)
        return self.controller(combined)


class IntegratedHierarchicalAgent:
    """整合的层次强化学习智能体"""
    
    def __init__(self, config):
        self.config = config
        
        # 网络
        self.net = HierarchicalDQN()
        self.target_net = HierarchicalDQN()
        self.target_net.load_state_dict(self.net.state_dict())
        
        # ✅ 使用AdamW和权重衰减
        self.optimizer = optim.AdamW(self.net.parameters(), lr=LR, weight_decay=1e-4)
    
        # ✅ 添加学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.9)
        self.critic = InternalCritic(config)

        # 底层执行模块
        self.rule_dqn_agent = RuleBasedDQNAgent(config)
        self.dispatch_heuristic = DispatchHeuristic()

        
        # 经验回放
        self.controller_buffer = deque(maxlen=100000)  # Controller经验
        self.meta_buffer = deque(maxlen=50000)        # Meta-Controller经验
        
        # 探索率管理
        self.epsilon_meta = INIT_EPSILON
        self.epsilon_controller = INIT_EPSILON
        self.steps_done = 0
        self.epsilon_g = {g: INIT_EPSILON for g in range(SUBGOAL_DIM)}  # ✅ 初始化子目标epsilon

        
        # 子目标跟踪
        self.subgoal_success = defaultdict(lambda: {'attempts': 0, 'successes': 0})
        self.current_subgoal = None
        self.subgoal_duration = 0
        self.max_subgoal_duration = 20

    def _build_features(self, state: Dict, config) -> torch.Tensor:
        """构建10维特征向量"""
        jobs = state['available_jobs']
        completed_jobs = state.get('completed_jobs', [])
        dispatched_jobs = state.get('dispatched_jobs', [])
        machines = state['machines']
        t = state.get('current_time', 0)

        total_jobs = len(jobs) + len(completed_jobs) + len(dispatched_jobs)
        completed_ratio = len(completed_jobs) / max(1, total_jobs)
        dispatched_ratio = len(dispatched_jobs) / max(1, total_jobs)
        avg_util = sum(getattr(m, 'utilization', 0.0) for m in machines) / max(1, len(machines))
        avg_wait = sum(getattr(j, 'waiting_time', 0.0) for j in jobs) / max(1, len(jobs))

        new_jobs = len(jobs) - len(completed_jobs)
        q_new = new_jobs / max(1, len(jobs))
        remaining_times = [float(m.remaining_time) for m in machines]
        sigma_mach_t = torch.tensor(remaining_times, dtype=torch.float32).std().item() / max(1e-6, config.max_processing_time)
        urgent_jobs = sum([1 for j in jobs if getattr(j, 'due_time', 1e9) <= t])
        urgency_ratio = urgent_jobs / len(jobs) if jobs else 0.0
        last_schedule_time = state.get('last_schedule_time', 0)
        schedule_age = (t - last_schedule_time) / max(1, config.max_processing_time)
        total_job_time = sum(getattr(j, 'processing_time', 0) for j in jobs)
        total_machine_capacity = sum(m.remaining_time for m in machines)
        system_pressure = total_job_time / max(1, total_machine_capacity)

        job_density = len(jobs) / max(1, len(machines))

        feats = torch.tensor([
            q_new,
            sigma_mach_t,
            urgency_ratio,
            schedule_age,
            system_pressure,
            completed_ratio,
            dispatched_ratio,
            avg_util,
            avg_wait,
            job_density
        ], dtype=torch.float32)
        
        if torch.isnan(feats).any() or torch.isinf(feats).any():
            feats = torch.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)
            
        return feats.unsqueeze(0)

    def _get_state(self, state: Dict) -> np.ndarray:
        """将环境状态转换为DQN输入状态 - 增强健壮性版本"""
        # 使用get方法安全地获取状态字段，提供默认值
        jobs = state.get('available_jobs', [])
        machines = state.get('machines', [])
        
        # 安全计算平均交期
        avg_due_date = 0.0
        if jobs:
            due_dates = [getattr(j, 'due_date', 0) for j in jobs if hasattr(j, 'due_date')]
            if due_dates:
                avg_due_date = np.mean(due_dates)
        
        # 安全计算平均处理时间
        proc_times = []
        if jobs:
            for j in jobs:
                if hasattr(j, 'operations'):
                    for op in j.operations:
                        if hasattr(op, 'processing_times'):
                            proc_times.extend(list(op.processing_times.values()))
        avg_proc_time = np.mean(proc_times) if proc_times else 0
        
        # 安全计算机器利用率
        machine_util = 0.0
        if machines:
            busy_count = sum(1 for m in machines if getattr(m, 'status', 'waiting') == 'busy')
            machine_util = busy_count / len(machines)
        
        # 安全获取其他状态字段
        current_time = state.get('current_time', 0)
        completed_jobs = state.get('completed_jobs', [])
        dispatched_jobs = state.get('dispatched_jobs', [])
        last_schedule_time = state.get('last_schedule_time', 0)
        last_batch_time = state.get('last_batch_time', 0)
        
        # 构建状态向量
        state_vec = np.array([
            len(jobs),
            len([j for j in jobs if getattr(j, 'status', 'waiting') == 'waiting']),
            avg_due_date,
            avg_proc_time,
            machine_util,
            current_time / max(1, self.config.max_time_steps),
            len(completed_jobs),
            len(dispatched_jobs),
            last_schedule_time / max(1, self.config.max_time_steps),
            last_batch_time / max(1, self.config.max_time_steps)
        ], dtype=np.float32)
        
        # 处理可能的NaN值
        state_vec = np.nan_to_num(state_vec, nan=0.0, posinf=1.0, neginf=0.0)
        
        return state_vec

    def _build_action_mask(self, state: Dict, subgoal: int) -> torch.Tensor:
        """构建动作掩码 - 根据子目标和状态有效性"""
        mask = torch.ones(ACTION_DIM, dtype=torch.bool)
        
        # 根据子目标类型调整掩码
        if subgoal == 0:  # 生产优化 - 偏向调度
            mask[8] = False  # 禁用配送
            mask[9] = False  # 禁用等待
        elif subgoal == 1:  # 配送效率 - 偏向配送
            mask[0:8] = False  # 禁用调度
            mask[9] = False    # 禁用等待
        # 系统平衡子目标允许所有动作
        
        # 通用有效性检查
        if not self._has_schedulable_jobs(state):
            mask[0:8] = False  # 无作业可调度
        if not self._has_dispatchable_jobs(state):
            mask[8] = False    # 无作业可配送
            
        return mask.unsqueeze(0)

    def _has_schedulable_jobs(self, state: Dict) -> bool:
        """检查是否有可调度的作业"""
        jobs = state['available_jobs']
        machines = state['machines']
        
        if not jobs:
            return False
            
        # 检查是否有等待状态的作业和空闲机器
        waiting_jobs = [j for j in jobs if getattr(j, 'status') == 'waiting']
        idle_machines = [m for m in machines if m.status == 'waiting']
        
        return len(waiting_jobs) > 0 and len(idle_machines) > 0

    def _has_dispatchable_jobs(self, state: Dict) -> bool:
        """检查是否有可配送的作业"""
        completed_jobs = state['completed_jobs']
        dispatched_jobs = state['dispatched_jobs']
        
        if not completed_jobs:
            return False
            
        # 检查是否有未配送的已完成作业
        completed_job_ids = {getattr(j, 'job_id', i) for i, j in enumerate(completed_jobs)}
        dispatched_job_ids = {getattr(j, 'job_id', i) for i, j in enumerate(dispatched_jobs)}
        
        return len(completed_job_ids - dispatched_job_ids) > 0

    def select_subgoal(self, state: Dict) -> int:
        """Meta-Controller选择子目标"""
        feats = self._build_features(state, self.config)
        
        # epsilon-greedy策略
        if random.random() < self.epsilon_meta:
            # 随机探索
            subgoal = random.randint(0, SUBGOAL_DIM - 1)
        else:
            # 贪婪选择
            with torch.no_grad():
                q_values = self.net.get_meta_q(feats)
                subgoal = q_values.argmax().item()
        
        # # 更新epsilon
        # self.epsilon_meta = max(FINAL_EPSILON, 
        #                       self.epsilon_meta * (EPS_ANNEAL_STEPS / NUM_EPISODES))
        
        return subgoal

    def select_action(self, state: Dict, subgoal: int) -> Dict:
        """Unified Controller选择具体动作"""
        feats = self._build_features(state, self.config)
        
        # 构建子目标one-hot编码
        subgoal_onehot = torch.zeros(1, SUBGOAL_DIM)
        subgoal_onehot[0, subgoal] = 1
        
        # 构建动作掩码
        mask = self._build_action_mask(state, subgoal)
        
         # 使用对应子目标的epsilon
        subgoal_epsilon = self.epsilon_g.get(subgoal, self.epsilon_controller)
        # epsilon-greedy策略
        if random.random() < subgoal_epsilon:
            # 随机探索，但只选择有效动作
            valid_actions = [i for i, m in enumerate(mask.squeeze()) if m]
            if valid_actions:
                action_idx = random.choice(valid_actions)
            else:
                action_idx = 9  # 默认等待
        else:
            # 贪婪选择
            with torch.no_grad():
                q_values = self.net.get_controller_q(feats, subgoal_onehot)
                masked_q_values = q_values.masked_fill(~mask, float('-inf'))
                action_idx = masked_q_values.argmax().item()
        
        # 执行对应的动作
        if action_idx < 8:  # 调度规则 (0-7)
            # 保存规则索引用于后续动作索引映射
            self.last_rule_idx = action_idx
            return self._apply_scheduling_rule(action_idx, state)
        elif action_idx == 8:  # 配送启发式
            return self.dispatch_heuristic.select_action(state)
        else:  # 等待动作 (9)
            return {'wait': True}

    def _apply_scheduling_rule(self, rule_idx: int, state: Dict) -> Dict[str, Dict[int, int]]:
        """应用指定的调度规则"""
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
        else:              # Random
            np.random.shuffle(jobs)
        
        # 分配作业到最早空闲机器
        schedule = {}
        assigned_machines = set()
    
        for job in jobs:
            op = job.operations[job.current_operation]
            for m_id in op.available_machine_ids:
                if machines[m_id].status == 'waiting' and m_id not in assigned_machines:
                    schedule[job.job_id] = m_id
                    assigned_machines.add(m_id)
                    break
        
        return {'schedule': schedule}

    def update_networks(self):
        """更新Controller和Meta-Controller网络"""
        
        # 更新Controller网络
        if len(self.controller_buffer) >= BATCH_SIZE:
            batch = random.sample(self.controller_buffer, BATCH_SIZE)
            states, subgoals, actions, rewards, next_states = zip(*batch)
            
            states_tensor = torch.FloatTensor(np.vstack(states))
            subgoals_tensor = torch.FloatTensor(np.vstack(subgoals))
            rewards_tensor = torch.FloatTensor(rewards)
            next_states_tensor = torch.FloatTensor(np.vstack(next_states))
            actions_tensor = torch.LongTensor(actions)
            
            # Controller网络更新
            current_q = self.net.get_controller_q(states_tensor, subgoals_tensor)
            current_q = current_q.gather(1, actions_tensor.unsqueeze(1))
            
            with torch.no_grad():
                next_q = self.target_net.get_controller_q(next_states_tensor, subgoals_tensor)
                next_q = next_q.max(1)[0]
                target = rewards_tensor + GAMMA * next_q
            
            loss = nn.MSELoss()(current_q.squeeze(), target)
            self.optimizer.zero_grad()
            loss.backward()
            # ✅ 添加梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        # 更新Meta-Controller网络
        if len(self.meta_buffer) >= BATCH_SIZE:
            batch = random.sample(self.meta_buffer, BATCH_SIZE)
            states, subgoals, F, next_states = zip(*batch)
            
            states_tensor = torch.FloatTensor(np.vstack(states))
            subgoals_tensor = torch.LongTensor(subgoals)
            F_tensor = torch.FloatTensor(F)
            next_states_tensor = torch.FloatTensor(np.vstack(next_states))
            
            # Meta-Controller网络更新
            current_q = self.net.get_meta_q(states_tensor)
            current_q = current_q[range(BATCH_SIZE), subgoals_tensor]
            
            with torch.no_grad():
                next_q = self.target_net.get_meta_q(next_states_tensor)
                next_q = next_q.max(1)[0]
                target = F_tensor + GAMMA * next_q
            
            loss = nn.MSELoss()(current_q, target)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        # ✅ 更频繁的软更新目标网络
        if self.steps_done % 100 == 0:
            tau = 0.01  # 软更新参数
            for target_param, param in zip(self.target_net.parameters(), self.net.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        self.scheduler.step()  # 更新学习率调度器
        self.optimizer.step()   # 优化器步骤

    def anneal_epsilon(self, episode):
        """退火探索率"""
        # Meta-Controller退火
        self.epsilon_meta = max(FINAL_EPSILON, INIT_EPSILON - 
                               (INIT_EPSILON - FINAL_EPSILON) * episode / NUM_EPISODES)
        
        # Controller自适应退火
        # for g in range(SUBGOAL_DIM):
        #     attempts = self.subgoal_success[g]['attempts']
        #     successes = self.subgoal_success[g]['successes']
        #     if attempts > 0:
        #         success_rate = successes / attempts
        #         self.epsilon_controller = max(FINAL_EPSILON, 1.0 - success_rate)
        # Controller退火 - 每个子目标独立的epsilon
        for g in range(SUBGOAL_DIM):
            attempts = self.subgoal_success[g]['attempts']
            successes = self.subgoal_success[g]['successes']
            if attempts > 50:  # 有足够数据后再调整
                success_rate = successes / attempts
                # 成功率越高，探索率越低
                self.epsilon_g[g] = max(FINAL_EPSILON, 0.3 - success_rate * 0.25)
            else:
                self.epsilon_g[g] = INIT_EPSILON

    def train_episode(self, env, episode):
        """训练一个完整的episode"""
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
            for _ in range(self.max_subgoal_duration):
                # 关键修复：使用Unified Controller网络选择动作
                action = self.select_action(state, subgoal)
                next_state, extrinsic, done, _ = env.step(action)
                intrinsic = self.critic.get_reward(next_state, subgoal)
                
                # 构建子目标one-hot编码
                subgoal_onehot = np.zeros(SUBGOAL_DIM)
                subgoal_onehot[subgoal] = 1
                
                # 确定动作索引
                action_idx = self._get_action_index(action)
                
                # 存储Controller经验 - 使用特征向量而不是原始字典
                state_features = self._get_state(state)
                next_state_features = self._get_state(next_state)
                self.controller_buffer.append((
                    state_features, subgoal_onehot, action_idx, intrinsic, next_state_features
                ))
                
                # 更新子目标成功统计
                self.subgoal_success[subgoal]['attempts'] += 1
                if intrinsic > 0:
                    self.subgoal_success[subgoal]['successes'] += 1
                    goal_achieved = True
                
                F += extrinsic
                total_extrinsic += extrinsic
                state = next_state
                episode_steps += 1
                self.steps_done += 1
                
                # 更新网络
                self.update_networks()
                
                if done or goal_achieved:
                    break
            
            # 存储Meta-Controller经验 - 使用特征向量而不是原始字典
            initial_state_features = self._get_state(initial_state)
            final_state_features = self._get_state(state)
            self.meta_buffer.append((
                initial_state_features, subgoal, F, final_state_features
            ))
            
            if done:
                break
        
        # 退火探索率
        self.anneal_epsilon(episode)
        return total_extrinsic

    def _get_action_index(self, action: Dict) -> int:
        """根据动作字典确定动作索引"""
        if 'schedule' in action:
            # 调度规则动作，返回规则索引
            return getattr(self, 'last_rule_idx', 0)
        elif 'dispatch' in action:
            # 配送动作
            return 8
        else:
            # 等待动作
            return 9


def run_integrated_hierarchical_dqn_experiment(config, case, seed):
    """运行整合的Hierarchical DQN实验"""
    set_random_seed(seed)
    config.seed = seed
    instance_id = generate_instance_id(config)
    env = WarehouseEnvironment(config, case)
    agent = IntegratedHierarchicalAgent(config)
    stats = defaultdict(list)
    
    episode_rewards = []
    start_time = time.time()
    
    for episode in range(NUM_EPISODES):
        reward = agent.train_episode(env, episode)
        print(f"Episode {episode}, Total Reward: {reward:.2f}, "
              f"Meta Epsilon: {agent.epsilon_meta:.3f}, "
              f"Controller Epsilon: {agent.epsilon_controller:.3f}, "
              f"Subgoal Success Rates: { {k: v['successes']/(v['attempts']+1e-5) for k, v in agent.subgoal_success.items()} }")
        episode_rewards.append(reward)
    
    # 计算总训练时间
    total_time = time.time() - start_time
    
    # 构建 stats 字典
    stats = {
        'episode_rewards': episode_rewards,
        'objective_value': -np.mean(episode_rewards),
        'algorithm_type': 'IntegratedHierarchicalDQN',
        'total_tardiness': getattr(env, 'total_tardiness', 0.0),
        'machine_utilization': getattr(env, 'machine_utilization', 0.0),
        'final_makespan': getattr(env, 'makespan', 0.0),
    }
    
    print(f"Training completed in {total_time:.2f} seconds.")
    print(f"All rewards: {episode_rewards}")
    
    # 调用保存函数存储实验结果
    save_algorithm_results_csv(
        algo_name="IntegratedHierarchicalDQN",
        instance_id=instance_id,
        seed=seed,
        config=config,
        stats=stats,
        env=env,
        total_time=total_time
    )


if __name__ == "__main__":
    config = Config()
    case = FlexibleJobShopScenario(config)
    run_integrated_hierarchical_dqn_experiment(config, case, seed=42)
