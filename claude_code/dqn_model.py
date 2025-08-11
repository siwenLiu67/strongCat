"""
使用深度Q网络(DQN)直接解决集成柔性作业车间调度与派遣问题(IFJSSP-DP)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import pickle
import time
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional

from case_generator import FlexibleJobShopScenario
from config import Config
from data_structures import Job, Operation, Machine


class DQNNetwork(nn.Module):
    """深度Q网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.FloatTensor(states), 
                torch.LongTensor(actions),
                torch.FloatTensor(rewards),
                torch.FloatTensor(next_states),
                torch.BoolTensor(dones))
    
    def __len__(self):
        return len(self.buffer)


class FJSSPEnvironment:
    """FJSSP-DP环境"""
    
    def __init__(self, scenario: FlexibleJobShopScenario):
        self.scenario = scenario
        self.jobs = scenario.jobs
        self.machines = scenario.machines
        self.distributors = scenario.distributors
        self.reset()
    
    def reset(self):
        """重置环境"""
        self.current_time = 0
        self.completed_jobs = []
        self.dispatched_jobs = []
        self.pending_jobs = self.jobs.copy()
        self.available_jobs = []
        
        # 重置机器状态
        for machine in self.machines:
            machine.status = "waiting"  # 使用data_structures中定义的状态
            machine.current_job = -1
            machine.remaining_time = 0.0
            machine.processed_jobs = []
            machine.total_busy_time = 0.0
            machine.total_idle_time = 0.0
        
        # 重置作业状态
        for job in self.jobs:
            job.status = "waiting"
            job.current_operation = 0  # 使用data_structures中的属性名
            job.completed_time = 0.0
            job.dispatched_time = 0.0
             
        self._update_available_jobs()
        return self._get_state()
    
    def _update_available_jobs(self):
        """更新可用作业列表"""
        # 检查新到达的作业
        for job in self.pending_jobs[:]:
            arrival_time = getattr(job, 'arrival_time', 0)
            if arrival_time <= self.current_time:
                self.available_jobs.append(job)
                self.pending_jobs.remove(job)
    
    def _get_state(self) -> np.ndarray:
        """获取当前状态向量"""
        state_features = []
        
        # 时间特征
        state_features.append(self.current_time / 100.0)  # 归一化时间
        
        # 机器状态特征
        idle_machines = sum(1 for m in self.machines if m.status == "waiting")
        busy_machines = len(self.machines) - idle_machines
        state_features.extend([
            idle_machines / len(self.machines),
            busy_machines / len(self.machines)
        ])
        
        # 作业特征
        total_jobs = len(self.jobs)
        waiting_jobs = sum(1 for j in self.available_jobs if j.status == "waiting")
        processing_jobs = sum(1 for j in self.jobs if j.status == "processing")
        completed_jobs = len(self.completed_jobs)
        dispatched_jobs = len(self.dispatched_jobs)
        
        state_features.extend([
            waiting_jobs / max(total_jobs, 1),
            processing_jobs / max(total_jobs, 1),
            completed_jobs / max(total_jobs, 1),
            dispatched_jobs / max(total_jobs, 1)
        ])
        
        # 紧急度特征
        if self.available_jobs:
            urgencies = []
            for job in self.available_jobs:
                # 计算剩余处理时间
                remaining_operations = job.operations[job.current_operation:]
                if remaining_operations:
                    remaining_time = sum(
                        min(op.processing_times.values()) if op.processing_times else 0
                        for op in remaining_operations
                    )
                else:
                    remaining_time = 0
                
                due_date = getattr(job, 'due_date', 100)
                urgency = max(0, (due_date - self.current_time - remaining_time)) / 100.0
                urgencies.append(urgency)
            
            state_features.extend([
                np.mean(urgencies),
                np.min(urgencies) if urgencies else 0,
                np.max(urgencies) if urgencies else 0
            ])
        else:
            state_features.extend([0, 0, 0])
        
        # 配送商负载特征
        for dist in self.distributors:
            dist_jobs = [j for j in self.jobs if j.distributor_id == dist.distributor_id]
            completed_for_dist = [j for j in dist_jobs if j in self.completed_jobs]
            load_ratio = len(completed_for_dist) / max(len(dist_jobs), 1)
            state_features.append(load_ratio)
        
        # 填充到固定长度
        target_length = 20
        while len(state_features) < target_length:
            state_features.append(0.0)
        
        return np.array(state_features[:target_length], dtype=np.float32)
    
    def _get_valid_actions(self) -> List[int]:
        """获取当前有效的动作"""
        valid_actions = []
        
        # 调度动作：为等待的作业分配机器
        for job in self.available_jobs:
            if job.status == "waiting" and job.current_operation < len(job.operations):
                current_op = job.operations[job.current_operation]
                for machine_id in current_op.available_machine_ids:
                    if machine_id < len(self.machines):
                        machine = self.machines[machine_id]
                        if machine.status == "waiting":  # 机器空闲状态
                            # 动作编码：job_id * 100 + machine_id
                            action_id = job.job_id * 100 + machine_id
                            valid_actions.append(action_id)
        
        # 派遣动作：为完成的作业创建批次
        completed_waiting = [j for j in self.completed_jobs 
                           if j not in self.dispatched_jobs]
        if completed_waiting:
            # 按配送商分组派遣
            for dist in self.distributors:
                dist_jobs = [j for j in completed_waiting 
                           if j.distributor_id == dist.distributor_id]
                if dist_jobs:
                    # 派遣动作编码：10000 + distributor_id
                    dispatch_action_id = 10000 + dist.distributor_id
                    valid_actions.append(dispatch_action_id)
        
        # 等待动作
        valid_actions.append(99999)  # 等待动作编码
        
        return valid_actions
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """执行动作"""
        reward = 0
        info = {}
        
        # 解析动作
        if action == 99999:
            # 等待动作
            reward = -0.1  # 等待惩罚
        elif action >= 10000:
            # 派遣动作
            distributor_id = action - 10000
            reward += self._dispatch_jobs(distributor_id)
        else:
            # 调度动作
            job_id = action // 100
            machine_id = action % 100
            reward += self._schedule_job(job_id, machine_id)
        
        # 推进时间
        self._advance_time()
        
        # 更新可用作业
        self._update_available_jobs()
        
        # 计算延误惩罚
        tardiness_penalty = self._calculate_tardiness_penalty()
        reward -= tardiness_penalty
        
        # 检查完成条件
        done = self._is_done()
        
        next_state = self._get_state()
        
        return next_state, reward, done, info
    
    def _schedule_job(self, job_id: int, machine_id: int) -> float:
        """调度作业到机器"""
        reward = 0
        
        # 找到对应的作业
        job = next((j for j in self.available_jobs if j.job_id == job_id), None)
        if not job or job.status != "waiting":
            return -1  # 无效动作惩罚
        
        if machine_id >= len(self.machines):
            return -1  # 机器ID无效
            
        machine = self.machines[machine_id]
        if machine.status != "waiting":
            return -1  # 机器忙碌惩罚
        
        # 获取当前工序
        if job.current_operation >= len(job.operations):
            return -1  # 工序已完成
            
        current_op = job.operations[job.current_operation]
        if machine_id not in current_op.available_machine_ids:
            return -1  # 机器不可用惩罚
        
        # 执行调度
        processing_time = current_op.processing_times.get(machine_id, 0)
        if processing_time <= 0:
            return -1  # 处理时间无效
            
        machine.assign_job(job_id, processing_time)  # 使用data_structures中的方法
        machine.status = "busy"
        
        job.status = "processing"
        if not hasattr(job, 'start_time') or job.start_time is None:
            job.start_time = self.current_time
        
        # 计算奖励
        due_date = getattr(job, 'due_date', 100)
        urgency = max(0, due_date - self.current_time) / max(due_date, 1)
        reward = 2.0 + urgency  # 基础调度奖励 + 紧急度奖励
        
        return reward
    
    def _dispatch_jobs(self, distributor_id: int) -> float:
        """派遣作业"""
        reward = 0
        
        # 找到该配送商的已完成未派遣作业
        completed_waiting = [j for j in self.completed_jobs 
                           if j not in self.dispatched_jobs 
                           and j.distributor_id == distributor_id]
        
        if not completed_waiting:
            return -0.5  # 无作业可派遣惩罚
        
        # 执行派遣
        for job in completed_waiting:
            job.dispatch_time = self.current_time
            job.dispatched_time = self.current_time  # 使用data_structures中的属性
            job.status = "dispatched"
            self.dispatched_jobs.append(job)
            
            # 计算派遣奖励
            due_date = getattr(job, 'due_date', 100)
            if job.completed_time <= due_date:
                reward += 1.0  # 按时完成奖励
            else:
                reward += 0.5  # 延误但完成奖励
        
        # 批次大小奖励
        batch_size = len(completed_waiting)
        reward += batch_size * 0.2
        
        return reward
    
    def _advance_time(self):
        """推进时间"""
        self.current_time += 1
        
        # 更新机器状态
        for machine in self.machines:
            if machine.status == "busy":
                machine.remaining_time -= 1
                if machine.remaining_time <= 0:
                    # 机器完成当前作业
                    job_id = machine.current_job
                    job = next((j for j in self.jobs if j.job_id == job_id), None)
                    
                    if job:
                        job.current_operation += 1  # 使用正确的属性名
                        if job.current_operation >= len(job.operations):
                            # 作业完成
                            job.status = "completed"
                            job.completed_time = self.current_time
                            if job not in self.completed_jobs:
                                self.completed_jobs.append(job)
                        else:
                            # 还有后续工序
                            job.status = "waiting"
                    
                    # 重置机器状态
                    machine.status = "waiting"  # 使用正确的状态值
                    machine.current_job = -1
                    machine.remaining_time = 0.0
    
    def _calculate_tardiness_penalty(self) -> float:
        """计算延误惩罚"""
        penalty = 0
        for distributor in self.distributors:
                min_due_time = min(distributor.delivery_requirements.due_times) if distributor.delivery_requirements else float('inf')
                if min_due_time < self.current_time:
                    # 计算每个配送商的延迟成本
                    requirement = distributor.delivery_requirements
                    for due_time, ratio, weight in zip(requirement.due_times, requirement.ratios, requirement.weights):
                        # 在这个due_time之前完成的作业
                        completed_jobs = [j for j in self.completed_jobs if j.dispatched_time <= due_time]
                        completed_amount = sum(j.amount for j in completed_jobs)
                        required_amount = ratio * distributor.total_amount
                        if completed_amount < required_amount:
                            penalty = (required_amount - completed_amount) * weight
                            reward -= int(penalty)
        # 计算配送完工时间延迟
        for job in self.completed_jobs:
            tardiness = max(0, job.dispatched_time - job.due_date)
            self.total_weighted_tardiness += tardiness
            reward -= tardiness 
                           
        return penalty+tardiness
    
    def _is_done(self) -> bool:
        """检查是否完成"""
        all_jobs_completed = len(self.completed_jobs) == len(self.jobs)
        all_jobs_dispatched = len(self.dispatched_jobs) == len(self.jobs)
        timeout = self.current_time > 500  # 超时限制
        
        return (all_jobs_completed and all_jobs_dispatched) or timeout


class DQNAgent:
    """DQN智能体"""
    
    def __init__(self, state_dim: int, action_dim: int, config: Config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # 网络参数
        self.lr = 0.001
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 10000
        self.target_update_freq = 100
        self.batch_size = 32
        self.memory_size = 10000
        
        # 神经网络
        self.q_net = DQNNetwork(state_dim, action_dim)
        self.target_q_net = DQNNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        
        # 经验回放
        self.memory = ReplayBuffer(self.memory_size)
        
        # 训练计数器
        self.steps = 0
        self.epsilon = self.epsilon_start
        
        # 更新目标网络
        self.target_q_net.load_state_dict(self.q_net.state_dict())
    
    def select_action(self, state: np.ndarray, valid_actions: List[int]) -> int:
        """选择动作"""
        self.steps += 1
        
        # 更新epsilon
        self.epsilon = max(self.epsilon_end, 
                          self.epsilon_start - (self.epsilon_start - self.epsilon_end) * 
                          self.steps / self.epsilon_decay)
        
        if random.random() < self.epsilon:
            # 随机探索
            return random.choice(valid_actions)
        else:
            # 贪婪选择
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_net(state_tensor)
                
                # 只考虑有效动作
                valid_q_values = []
                for action in valid_actions:
                    if action < self.action_dim:
                        valid_q_values.append((q_values[0][action].item(), action))
                    else:
                        # 对于超出范围的动作使用默认值
                        valid_q_values.append((0.0, action))
                
                # 选择Q值最大的动作
                best_action = max(valid_q_values, key=lambda x: x[0])[1]
                return best_action
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储经验"""
        # 将动作映射到有效范围
        action_idx = min(action, self.action_dim - 1)
        self.memory.push(state, action_idx, reward, next_state, done)
    
    def update(self):
        """更新网络"""
        if len(self.memory) < self.batch_size:
            return 0
        
        # 采样批次
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # 计算当前Q值
        current_q_values = self.q_net(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_q_net(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # 计算损失
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        if self.steps % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        return loss.item()

import csv, time
from pathlib import Path
import numpy as np

def save_results_csv(algo_name, instance_id, seed, config, stats, env, total_time):
    """
    统一保存实验结果（精简版，与论文表格对齐）。
    必备环境字段：
      - job: completed_time, due_date, dispatched_time
      - distributor: distributor_id, required_jobs(可选), weight(可选)
    """
    output_file = Path("results") / "summary_table.csv"
    output_file.parent.mkdir(exist_ok=True)

    # 运行效率
    num_steps = int(sum(stats.get('episode_lengths', [])))
    time_per_step = (total_time / num_steps * 1000.0) if num_steps > 0 else 0.0

    # —— 核心指标 —— #
    # 1) 总延迟 ∑ T_j
    total_tardiness = 0.0
    for job in getattr(env, 'jobs', []):
        due = float(getattr(job, 'due_date', 0.0))
        c   = float(getattr(job, 'completed_time', 0.0))
        total_tardiness += max(0.0, c - due)

    # 2) 加权短缺 ∑ w_r u_r
    #    若缺 required_jobs 或 weight，使用默认值：required_jobs=0, weight=0（不计入罚）
    weighted_shortage = 0.0
    for dist in getattr(env, 'distributors', []):
        did = getattr(dist, 'distributor_id', None)
        required = int(getattr(dist, 'required_jobs', 0))
        weight   = float(getattr(dist, 'weight', 0.0))
        delivered = sum(1 for j in getattr(env, 'dispatched_jobs', [])
                        if getattr(j, 'distributor_id', None) == did)
        shortage = max(0, required - delivered)
        weighted_shortage += weight * shortage

    # 3) 目标值
    objective_sum = total_tardiness + weighted_shortage

    # 4) 辅助指标
    jobs_list = getattr(env, 'jobs', [])
    makespan = float(getattr(env, 'current_time', 0.0))

    on_time_rate = (np.mean([
        1.0 if float(getattr(j, 'completed_time', 0.0)) <= float(getattr(j, 'due_date', 0.0)) else 0.0
        for j in jobs_list
    ]) if jobs_list else 0.0)

    # 需求覆盖率（按配送商平均覆盖）
    cover_vals = []
    for dist in getattr(env, 'distributors', []):
        required = int(getattr(dist, 'required_jobs', 0))
        if required <= 0:
            continue
        did = getattr(dist, 'distributor_id', None)
        delivered = sum(1 for j in getattr(env, 'dispatched_jobs', [])
                        if getattr(j, 'distributor_id', None) == did)
        cover_vals.append(min(1.0, delivered / required))
    req_coverage = float(np.mean(cover_vals)) if cover_vals else 0.0

    # 平均派遣延迟
    late_dispatch = (np.mean([
        max(0.0, float(getattr(j, 'dispatched_time', 0.0)) - float(getattr(j, 'due_date', 0.0)))
        for j in jobs_list
    ]) if jobs_list else 0.0)

    # 行数据（字段名固定，便于后续统一读表）
    row = {
        "run_id": f"{time.strftime('%Y%m%d')}_{seed}",
        "algo": algo_name,
        "instance_id": instance_id,
        "seed": int(seed),
        "num_jobs": len(jobs_list),
        "num_machines": len(getattr(env, 'machines', [])),
        "num_distributors": len(getattr(env, 'distributors', [])),
        "episodes": int(getattr(config, 'episodes', 0)),
        "total_tardiness": float(total_tardiness),
        "weighted_shortage": float(weighted_shortage),
        "objective_sum": float(objective_sum),
        "makespan": float(makespan),
        "on_time_rate": float(on_time_rate),
        "req_coverage": float(req_coverage),
        "late_dispatch": float(late_dispatch),
        "time_per_step_ms": float(time_per_step),
    }

    file_exists = output_file.exists()
    with open(output_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def main():
    """主训练函数"""
    # 配置参数
    config = Config()
    config.num_initial_jobs = 8
    config.num_machines = 4
    config.num_distributors = 2
    config.min_operations = 2
    config.max_operations = 4
    
    print("生成FJSP-DP场景...")
    scenario = FlexibleJobShopScenario(config=config)
    
    print(f"场景信息:")
    print(f"- 作业数量: {len(scenario.jobs)}")
    print(f"- 机器数量: {len(scenario.machines)}")
    print(f"- 配送商数量: {len(scenario.distributors)}")
    
    # 创建环境和智能体
    env = FJSSPEnvironment(scenario)
    state_dim = 20  # 状态维度
    action_dim = 20000  # 动作空间大小（包含所有可能的调度和派遣动作）
    
    agent = DQNAgent(state_dim, action_dim, config)
    
    # 训练参数
    episodes = 200
    stats = defaultdict(list) 
    
    print(f"\n开始DQN训练 {episodes} episodes...")
    start_time = time.time()
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            # 获取有效动作
            valid_actions = env._get_valid_actions()
            if not valid_actions:
                valid_actions = [99999]  # 至少包含等待动作
            
            # 选择动作
            action = agent.select_action(state, valid_actions)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            agent.store_transition(state, action, reward, next_state, done)
            
            # 更新网络
            loss = agent.update()
            
            # 更新状态和统计
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        # 记录统计数据
        stats['episode_rewards'].append(episode_reward)
        stats['episode_lengths'].append(episode_length)
        stats['makespans'].append(env.current_time)
        stats['completed_jobs'].append(len(env.completed_jobs))
        stats['dispatched_jobs'].append(len(env.dispatched_jobs))
        stats['epsilon'].append(agent.epsilon)
        
        # 打印进度
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(stats['episode_rewards'][-20:])
            avg_makespan = np.mean(stats['makespans'][-20:])
            print(f"Episode {episode + 1}/{episodes}")
            print(f"  平均奖励: {avg_reward:.2f}")
            print(f"  平均makespan: {avg_makespan:.2f}")
            print(f"  完成作业: {len(env.completed_jobs)}/{len(env.jobs)}")
            print(f"  派遣作业: {len(env.dispatched_jobs)}/{len(env.jobs)}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
    
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
    
    with open('dqn_results.pkl', 'wb') as f:
        pickle.dump(result_data, f)
    
    # 保存模型
    torch.save(agent.q_net.state_dict(), 'dqn_model.pth')
    
    print(f"\n训练完成！")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"平均奖励: {np.mean(stats['episode_rewards']):.2f}")
    print(f"平均makespan: {np.mean(stats['makespans']):.2f}")
    print(f"平均完成作业数: {np.mean(stats['completed_jobs']):.2f}")
    print(f"平均派遣作业数: {np.mean(stats['dispatched_jobs']):.2f}")
    print(f"结果已保存至: dqn_results.pkl")
    print(f"模型已保存至: dqn_model.pth")

    # 统计总步数（用于 time_per_step）
    steps_sum = int(np.sum(stats['episode_lengths'])) if len(stats['episode_lengths']) > 0 else 0
    # 写入CSV（确保 config 有 episodes 字段）
    config.episodes = episodes
    save_results_csv(
        algo_name="DQN",
        instance_id="Small-01",   # ← 替换为你的实例命名
        seed=getattr(config, 'seed', 42),
        config=config,
        stats=stats,
        env=env,
        total_time=total_time
    )


def run_dqn_experiment(config, case, seed, **kwargs):
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
    action_dim = 20000
    agent = DQNAgent(state_dim, action_dim, config)
    episodes = getattr(config, "episodes", 50)
    stats = defaultdict(list)

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        while True:
            valid_actions = env._get_valid_actions()
            if not valid_actions:
                valid_actions = [99999]
            action = agent.select_action(state, valid_actions)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            episode_reward += reward
            if done:
                break
        stats['episode_rewards'].append(episode_reward)
        stats['makespans'].append(env.current_time)
        stats['completed_jobs'].append(len(env.completed_jobs))
        stats['dispatched_jobs'].append(len(env.dispatched_jobs))

    result = {
        "stats": stats,
        "env": env,
        "additional_metrics": {}
    }
    return result

def test_trained_model():
    """测试训练好的模型"""
    config = Config()
    config.num_initial_jobs = 6
    config.num_machines = 3
    config.num_distributors = 2
    
    # 生成测试场景
    scenario = FlexibleJobShopScenario(config=config)
    env = FJSSPEnvironment(scenario)
    
    # 加载训练好的模型
    state_dim = 20
    action_dim = 20000
    agent = DQNAgent(state_dim, action_dim, config)
    agent.q_net.load_state_dict(torch.load('dqn_model.pth'))
    agent.epsilon = 0  # 禁用探索
    
    print("测试训练好的DQN模型...")
    
    state = env.reset()
    total_reward = 0
    
    while True:
        valid_actions = env._get_valid_actions()
        if not valid_actions:
            valid_actions = [99999]
        
        action = agent.select_action(state, valid_actions)
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


if __name__ == "__main__":
    main()
    
    # 取消注释以测试训练好的模型
    # test_trained_model()