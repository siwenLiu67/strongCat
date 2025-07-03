import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List
import torch
from environment import WarehouseEnvironment

class HighLevelAgent:
    """元控制器：
    负责在更高层次上做决策：
    0: 生产调度
    1: 订单配送
    2: 等待观察
    """
    def __init__(self, config):
        self.config = config
        self._build_network()
        self._setup_training()
        self.last_action = None

    def _build_features(self, state: Dict) -> torch.Tensor:
        """
        特征包含:
            1. q_new: 新到达作业占当前总作业数的比例
            2. sigma_mach_t: 机器负载不平衡程度
            3. urgency_ratio: 紧急程度
            4. schedule_age: 当前调度方案的年龄
            5. system_pressure: 系统压力指标
        """
        # 1. q_new
        jobs = state['available_jobs']
        new_jobs = len(state['available_jobs']) - len(state['completed_jobs'])
        q_new = new_jobs / max(1, len(jobs))
        
        # 2. sigma_mach_t
        machines = state['machines']
        remaining_times = [float(m.remaining_time) for m in machines]
        sigma_mach_t = torch.tensor(remaining_times, dtype=torch.float32).std().item() / max(1e-6, self.config.max_processing_time)
        
        # 3. urgency_ratio
        # 假设每个job有due_time与current_time，统计逾期的比例
        urgent_jobs = 0
        if jobs:
            urgent_jobs = sum([1 for j in jobs if getattr(j, 'due_time', 1e9) <= state['current_time']])
            urgency_ratio = urgent_jobs / len(jobs)
        else:
            urgency_ratio = 0.0
        
        # 4. schedule_age
        current_time = state['current_time']
        last_schedule_time = state.get('last_schedule_time', 0)
        schedule_age = (current_time - last_schedule_time) / max(1, self.config.max_processing_time)
        
        # 5. system_pressure
        total_job_time = sum(getattr(j, 'processing_time', 0) for j in jobs)
        total_machine_capacity = sum(m.remaining_time for m in machines)
        system_pressure = total_job_time / max(1, total_machine_capacity)
        
        feats = torch.tensor([
            q_new,
            sigma_mach_t,
            urgency_ratio,
            schedule_age,
            system_pressure
        ], dtype=torch.float32)
        return feats.unsqueeze(0)

    def _get_action_mask(self, state: Dict) -> torch.Tensor:
        mask = torch.ones(3, dtype=torch.bool)
        if sum(state.get('inventory_levels', [])) == 0:
            mask[1] = False
        if all(m.remaining_time > 0 for m in state['machines']):
            mask[0] = False
        return mask.unsqueeze(0)

    def _build_network(self):
        self.input_dim = 5
        self.hidden_dim = 32
        self.action_dim = 3
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=2, batch_first=True)
        self.fc2 = nn.Linear(self.hidden_dim, self.action_dim)
        # ——初始化最后一层bias，让三个动作初始输出完全一样
        with torch.no_grad():
            self.fc2.bias[0] = 0.5   # 调度
            self.fc2.bias[1] = 0.5   # 配送
            self.fc2.bias[2] = -0.2  # 等待

        nn.init.constant_(self.fc2.bias, 0.)

    def _setup_training(self):
        self.optimizer = torch.optim.Adam(
            list(self.fc1.parameters()) +
            list(self.attn.parameters()) +
            list(self.fc2.parameters()), lr=1e-3)

    def select_action(self, state: Dict) -> Tuple[int, torch.Tensor]:
        feats = self._build_features(state)
        mask = self._get_action_mask(state)
        x = F.relu(self.fc1(feats)).unsqueeze(1)
        attn_out, _ = self.attn(x, x, x)
        x = attn_out.squeeze(1)
        logits = self.fc2(x)
        logits = logits.masked_fill(~mask, float('-inf'))
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        self.last_action = int(action.item())
        return int(action.item()), log_prob

    def compute_reward(self, state: Dict) -> float:
        delay = state.get('total_delay', 0)
        p_delay = -self.config.delay_penalty * delay
        inv = sum(state.get('inventory_levels', []))
        p_inventory = -self.config.inventory_penalty * (inv / max(1, self.config.max_inventory))
        p_stability = self.config.stability_reward if self.last_action == state.get('prev_action') else 0
        return p_delay + p_inventory + p_stability

    def update(self, batch: Dict):
        """基于 REINFORCE 算法更新策略网络"""
        # batch 包含：states, actions, log_probs, returns
        states: List[Dict] = batch['states']
        actions = torch.tensor(batch['actions'], dtype=torch.int64)
        old_log_probs = torch.stack(batch['log_probs'])  # [T]
        returns = torch.tensor(batch['returns'], dtype=torch.float32)  # [T]

        # 重新计算 log_prob
        new_log_probs = []
        for state, action in zip(states, actions):
            _, log_prob = self.select_action(state)
            new_log_probs.append(log_prob)
        new_log_probs = torch.stack(new_log_probs)

        # 计算损失：- sum (log_prob * return)
        loss = -torch.sum(new_log_probs * returns)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
