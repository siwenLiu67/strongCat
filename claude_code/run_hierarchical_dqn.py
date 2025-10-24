import time
import numpy as np
from collections import defaultdict
import torch
import torch.optim as optim
from collections import defaultdict, deque
from environment import WarehouseEnvironment
from rule_based_agent import RuleBasedDQNAgent 
from dispatch_heuristic import DispatchHeuristic
from high_level_dqn_agent import HighLevelDQNAgent  # 替换为DQN智能体
from case_generator import FlexibleJobShopScenario
from config import Config
from typing import Dict, List, Tuple
from algorithm_results_saver import save_algorithm_results_csv, generate_instance_id, set_random_seed
import random
import torch.nn as nn

# 环境参数
STATE_DIM = 10  # 实际特征维度
SUBGOAL_DIM = 3
ACTION_DIM = 8  # 8种调度规则


# 训练参数
NUM_EPISODES = 20
INIT_EPSILON = 0.9
FINAL_EPSILON = 0.05
EPS_ANNEAL_STEPS = NUM_EPISODES*0.8
BATCH_SIZE = 128
GAMMA = 0.99
LR = 0.00005

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
        # 关键指标：机器利用率、作业完成率、调度效率
      #  machine_utilization = state['machine_utilization']
        completed_jobs = len(state['completed_jobs'])
      #  active_jobs = len(state['available_jobs'])
        
        # 计算生产效率得分
      #  utilization_score = min(1.0, machine_utilization / 0.8)  # 目标80%利用率
        completion_rate = completed_jobs / (self.config.num_initial_jobs + self.config.num_dynamic_jobs)
        
        # 综合生产得分
     #   production_score = (utilization_score * 0.6 + # completion_rate * 0.4)
        return completion_rate  # 放大奖励
    
    def _evaluate_delivery_goal(self, state):
        dispatched_jobs = state['dispatched_jobs']
        if not dispatched_jobs:
            return 0.0
        total_tardiness = sum(max(0, j.dispatched_time - j.due_date) for j in dispatched_jobs)
        avg_tardiness = total_tardiness / len(dispatched_jobs)
        return 1.0 / (1.0 + avg_tardiness)  # 小延迟 -> 奖励接近1

    
    def _evaluate_balancing_goal(self, state):
        """评估系统平衡子目标达成度"""
        # 关键指标：系统负载均衡、资源分配、避免瓶颈
        return state['machine_utilization']   # 平衡目标奖励相对较小


class HierarchicalDQN(nn.Module):
    def __init__(self):
        super().__init__()
        # Meta-Controller网络
        self.meta_controller = nn.Sequential(
            nn.Linear(STATE_DIM, 64),  # 修正为实际状态维度
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
        # Controller网络（目标条件策略）
        self.controller = nn.Sequential(
            nn.Linear(STATE_DIM + SUBGOAL_DIM, 64),  # 修正为13维(10+3)
            nn.ReLU(),
            nn.Linear(64, ACTION_DIM)
        )
    
    def forward(self, state):
        """前向传播 - 返回Meta-Controller的Q值"""
        return self.meta_controller(state)
    
    def get_meta_q(self, state):
        return self.meta_controller(state)
    
    def get_controller_q(self, state, subgoal):
        combined = torch.cat([state, subgoal], dim=1)
        return self.controller(combined)

    
class HierarchicalAgent:
    def __init__(self, config):
        self.config = config  # 存储config参数
        self.net = HierarchicalDQN()
        self.target_net = HierarchicalDQN()
        self.target_net.load_state_dict(self.net.state_dict())
        
        self.optimizer = optim.Adam(self.net.parameters(), lr=LR)
        self.critic = InternalCritic(config)

        # 初始化底层智能体
        self.rule_dqn_agent = RuleBasedDQNAgent(config)
        self.dispatch_agent = DispatchHeuristic()
        
        # 经验回放
        self.D1 = deque(maxlen=100000)  # Controller经验
        self.D2 = deque(maxlen=50000)   # Meta-Controller经验
        self.state_dim = 10  # 状态维度
        self.action_dim = 8  # 8种调度规则
        # 探索率管理
        self.epsilon_g = {g: INIT_EPSILON for g in range(SUBGOAL_DIM)}
        self.epsilon_meta = INIT_EPSILON
        self.steps_done = 0
        self.epsilon_start = 0.99
        self.epsilon_end = 0.90
        self.epsilon_decay = 10000
        self.count = 0
        self.loss_list = []
        self.meta_loss_list = []
        
        # 子目标跟踪
        self.subgoal_success = defaultdict(lambda: {'attempts': 0, 'successes': 0})

    def _build_features(self, state: Dict, config) -> torch.Tensor:
        """构建10维特征向量以匹配网络输入维度"""
        jobs = state['available_jobs']
        completed_jobs = state.get('completed_jobs', [])
        dispatched_jobs = state.get('dispatched_jobs', [])
        machines = state['machines']
        t = state.get('current_time', 0)

        # 新特征
        total_jobs = len(jobs) + len(completed_jobs) + len(dispatched_jobs)
        completed_ratio = len(completed_jobs) / max(1, total_jobs)
        dispatched_ratio = len(dispatched_jobs) / max(1, total_jobs)
        avg_util = sum(getattr(m, 'utilization', 0.0) for m in machines) / max(1, len(machines))
        avg_wait = sum(getattr(j, 'waiting_time', 0.0) for j in jobs) / max(1, len(jobs))

        # 原有特征
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

        # 添加第10个特征：作业密度
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
        
        # 检查特征值是否有效
        if torch.isnan(feats).any() or torch.isinf(feats).any():
            feats = torch.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)
            
        return feats.unsqueeze(0)

    def _get_action_mask(self, state: Dict) -> torch.Tensor:
        """复用现有的动作掩码方法"""
        mask = torch.ones(3, dtype=torch.bool)
        completed_jobs = state.get('completed_jobs', [])
        dispatched_jobs = state.get('dispatched_jobs', [])
        
        # 修复：检查是否有未配送的已完成作业
        completed_job_ids = {getattr(j, 'job_id', i) for i, j in enumerate(completed_jobs)}
        dispatched_job_ids = {getattr(j, 'job_id', i) for i, j in enumerate(dispatched_jobs)}
        
        # 如果没有未配送的已完成作业，禁用配送动作
        if not completed_job_ids or len(completed_job_ids - dispatched_job_ids) == 0:
            mask[1] = False

        # 如果都在配送中状态，禁用配送动作
        if all(getattr(j, 'status') in ['dispatching', 'dispatched'] for j in state['available_jobs']):
            mask[1] = False
       
        # 如果没有可调度作业，禁用调度动作
        jobs = state.get('available_jobs', [])
        if not jobs or not any(getattr(j, 'status', None) == 'waiting' for j in jobs):
            mask[0] = False

        
        # 只要有空闲机器且有可调度作业，就允许schedule
        if not state['machines'] or all(m.remaining_time > 0 for m in state['machines']):
            mask[0] = False
            
        return mask.unsqueeze(0)

    def _get_state(self, state: Dict) -> np.ndarray:
        """将环境状态转换为DQN输入状态"""
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
        
        current_time = state['current_time']
        completed_jobs = state['completed_jobs']
        dispatched_jobs = state['dispatched_jobs']
        last_schedule_time = state['last_schedule_time']
        last_batch_time = state['last_batch_time']
        
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
    
    def select_action(self, state, subgoal):
        """Controller的epsilon-greedy策略 - 最小改动集成"""
        if subgoal == 0:
            # 使用RuleBasedDQNAgent进行生产调度
            action, rule_idx = self.rule_dqn_agent.select_action(state)
            self.last_rule_idx = rule_idx
            return action

        elif subgoal == 1:
            # 使用DispatchHeuristic进行配送决策
            action = self.dispatch_agent.select_action(state)
            self.last_rule_idx = 7  # 假设7代表配送规则
            return action
        
        else:
            # 等待动作
            return {'wait': True}
        
            
    def _apply_rule(self, rule_idx: int, state: Dict) -> Dict[str, Dict[int, int]]:
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
        else:              # Random
            np.random.shuffle(jobs)
        
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
                else:
                    # 如果当前机器已分配作业，跳过该机器
                    print(f"Machine {m_id} is already assigned, skipping.")
        
        return {'schedule': schedule}


    def select_subgoal(self, state) -> int:
        """Meta-Controller的epsilon-greedy策略"""
        """选择动作 - 使用DQN的ε-贪婪策略"""

        # 构建特征
        feats = self._build_features(state, self.config)
        # 获取动作掩码
        mask = self._get_action_mask(state)
        
        # ε-贪婪策略
        if random.random() < self.epsilon_meta:
            # 随机探索，但只选择有效动作
            valid_actions = [i for i, m in enumerate(mask.squeeze()) if m]
            if valid_actions:
                action = random.choice(valid_actions)
            else:
                action = 2  # 默认等待
        else:
            # 贪婪选择
            with torch.no_grad():
                q_values = self.net(feats)
                
                # 应用动作掩码
                masked_q_values = q_values.masked_fill(~mask, float('-inf'))
                
                # 选择Q值最大的动作
                action = masked_q_values.argmax().item()
        
        # 返回动作和占位符（保持接口兼容）
        self.last_action = action
        return action

    def update_networks(self):
        # 更新Controller
        if len(self.D1) >= BATCH_SIZE:
            batch = random.sample(self.D1, BATCH_SIZE)
            states, subgoals, actions, rewards, next_states = zip(*batch)
            
            state_features = torch.stack([self._build_features(state, self.config).squeeze(0) for state in states])
            next_state_features = torch.stack([self._build_features(state, self.config).squeeze(0) for state in next_states])
            
            subgoals_tensor = torch.FloatTensor(np.vstack(subgoals))
            rewards_tensor = torch.FloatTensor(rewards)
            
            actions_tensor = torch.LongTensor(actions)
            
            current_q = self.net.get_controller_q(state_features, subgoals_tensor).gather(1, actions_tensor.unsqueeze(1))
            
            with torch.no_grad():
                next_q = self.target_net.get_controller_q(next_state_features, subgoals_tensor).max(1)[0]
                target = rewards_tensor + GAMMA * next_q
            
            loss = nn.MSELoss()(current_q.squeeze(), target)
            self.loss_list.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # 更新Meta-Controller
        if len(self.D2) >= BATCH_SIZE:
            batch = random.sample(self.D2, BATCH_SIZE)
            states, subgoals, F, next_states = zip(*batch)
            
            # 将状态字典转换为特征向量
            state_features = [self._get_state(state) for state in states]
            next_state_features = [self._get_state(state) for state in next_states]
            
            states = torch.FloatTensor(np.vstack(state_features))
            subgoals = torch.LongTensor(subgoals)
            F = torch.FloatTensor(F)
            next_states = torch.FloatTensor(np.vstack(next_state_features))
            
            current_q = self.net.get_meta_q(states)[range(BATCH_SIZE), subgoals]
            with torch.no_grad():
                next_q = self.target_net.get_meta_q(next_states).max(1)[0]
                target = F + GAMMA * next_q
            
            loss = nn.MSELoss()(current_q, target)
            self.meta_loss_list.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # 更新目标网络
        if self.count % 20 == 0:
            self.target_net.load_state_dict(self.net.state_dict())
        self.count += 1

    def anneal_epsilon(self, episode):
        """标准的epsilon退火策略"""
        
        # 1. Meta-Controller: 标准指数退火
        if EPS_ANNEAL_STEPS > 0:
            # 线性衰减到指定步数，然后保持最小值
            progress = min(1.0, episode / EPS_ANNEAL_STEPS)
            self.epsilon_meta = INIT_EPSILON - (INIT_EPSILON - FINAL_EPSILON) * progress

        else:
            # 基于总episode数的指数退火
            decay_rate = (FINAL_EPSILON / INIT_EPSILON) ** (1.0 / max(1, NUM_EPISODES))
            self.epsilon_meta = max(FINAL_EPSILON, INIT_EPSILON * (decay_rate ** episode))
        
        # 2. Controller: 基于成功率的自适应退火
        for g in range(SUBGOAL_DIM):
            attempts = self.subgoal_success[g]['attempts']
            successes = self.subgoal_success[g]['successes']
            
            if attempts >= 10:  # 有足够统计数据
                success_rate = successes / attempts
                
                # 修复：保持合理探索，即使成功率很高
                base_epsilon = 0.1  # 基础探索率
                adaptive_component = (1.0 - success_rate) * 0.3  # 自适应部分
                
                self.epsilon_g[g] = max(FINAL_EPSILON, base_epsilon + adaptive_component)
            else:
                # 数据不足时使用中等探索率
                self.epsilon_g[g] = 0.3

    def train_episode(self, env, episode):
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
                # 然后在经验回放中存储动作索引
                self.D1.append((
                    state, subgoal_onehot, self.last_rule_idx, intrinsic, next_state  # 存储动作索引而不是动作对象
                ))
                
                # 更新统计
                self.subgoal_success[subgoal]['attempts'] += 1
                if intrinsic > 0:
                    self.subgoal_success[subgoal]['successes'] += 1
                    goal_achieved = True
                
                F += extrinsic
                total_extrinsic += extrinsic  # 累加总外在奖励
                state = next_state
                episode_steps += 1
                self.steps_done += 1
                
                # 更新网络
                if self.steps_done % 10 == 0:
                    self.update_networks()
                
                if done or goal_achieved:
                    break
            
            # 存储Meta-Controller经验
            self.D2.append((
                initial_state, subgoal, F, state
            ))
            
            if done:
                break

            print(f"Episode {episode}, Step {episode_steps}, Subgoal {subgoal}, "
                  f"Extrinsic Reward: {F:.2f}, Total Extrinsic: {total_extrinsic:.2f}, ")
        
        # 退火探索率
        self.anneal_epsilon(episode)
        return total_extrinsic

def run_hierarchical_dqn_experiment(config, case, seed):
    set_random_seed(seed)
    config.seed = seed
    instance_id = generate_instance_id(config)
    env = WarehouseEnvironment(config, case)
    agent = HierarchicalAgent(config)
    stats = defaultdict(list)
    
    episode_rewards = []
    start_time = time.time()
    
    for episode in range(NUM_EPISODES):
        reward = agent.train_episode(env, episode)
        print(f"Episode {episode}, Total Reward: {reward:.2f}, "
          f"Meta Epsilon: {agent.epsilon_meta:.3f}, "
          f"Subgoal Success Rates: { {k: v['successes']/(v['attempts']+1e-5) for k, v in agent.subgoal_success.items()} }",
          f"total penalty: {env.tardy_penalty},",
          f"tardiness: {env.total_weighted_tardiness},",
          f"makespan: {env.t},")
        episode_rewards.append(reward)
    
    # 计算总训练时间
    total_time = time.time() - start_time
    
    # 构建 stats 字典，包含所有需要的指标
    stats = {
        'episode_rewards': episode_rewards,
        'objective_value': -np.mean(episode_rewards),
        'algorithm_type': 'HierarchicalDQN',
        'total_tardiness': getattr(env, 'total_tardiness', 0.0),
        'machine_utilization': getattr(env, 'machine_utilization', 0.0),
        'final_makespan': getattr(env, 'makespan', 0.0),
    }
    
    print(f"Training completed in {total_time:.2f} seconds.")
    print(f"all rewards: {episode_rewards}")
    print(f"all sub losses: {agent.loss_list}")
    print(f"all meta losses: {agent.meta_loss_list}")

    
    # 调用保存函数存储实验结果
    save_algorithm_results_csv(
        algo_name="HierarchicalDQN",
        instance_id=instance_id,
        seed=seed,
        config=config,
        stats=stats,
        env=env,
        total_time=total_time
    )
        
if __name__ == "__main__":
    config = Config()
    config.max_operations = 2
    config.num_initial_jobs = 4
    config.num_dynamic_jobs = 1
    config.max_time_steps = 50
    config.num_distributors = 2
   
    case = FlexibleJobShopScenario(config)
    run_hierarchical_dqn_experiment(config, case, seed=42)
