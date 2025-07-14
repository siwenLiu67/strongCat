import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List
from environment import WarehouseEnvironment

class HighLevelAgent(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._build_network()
        self._setup_training()
        self.last_action = None

    def _build_features(self, state: Dict) -> torch.Tensor:
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
        sigma_mach_t = torch.tensor(remaining_times, dtype=torch.float32).std().item() / max(1e-6, self.config.max_processing_time)
        urgent_jobs = sum([1 for j in jobs if getattr(j, 'due_time', 1e9) <= t])
        urgency_ratio = urgent_jobs / len(jobs) if jobs else 0.0
        last_schedule_time = state.get('last_schedule_time', 0)
        schedule_age = (t - last_schedule_time) / max(1, self.config.max_processing_time)
        total_job_time = sum(getattr(j, 'processing_time', 0) for j in jobs)
        total_machine_capacity = sum(m.remaining_time for m in machines)
        system_pressure = total_job_time / max(1, total_machine_capacity)

        feats = torch.tensor([
            q_new,
            sigma_mach_t,
            urgency_ratio,
            schedule_age,
            system_pressure,
            completed_ratio,
            dispatched_ratio,
            avg_util,
            avg_wait
        ], dtype=torch.float32)
        
        # 检查特征值是否有效
        if torch.isnan(feats).any() or torch.isinf(feats).any():
            feats = torch.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)
            
        return feats.unsqueeze(0)

    def _get_action_mask(self, state: Dict) -> torch.Tensor:
        mask = torch.ones(3, dtype=torch.bool)
        completed_jobs = state.get('completed_jobs', [])
        dispatched_jobs = state.get('dispatched_jobs', [])
        # 只要有未配送的已完成作业，就允许dispatch
        if not any(j for j in completed_jobs if j not in dispatched_jobs):
            mask[1] = False
        # 只要有空闲机器且有可调度作业，就允许schedule
        if not any(m.remaining_time == 0 for m in state['machines']) or not state['available_jobs']:
            mask[0] = False
        return mask.unsqueeze(0)

    def _build_network(self):
        self.input_dim = 9  # 特征数
        self.hidden_dim = 32
        self.action_dim = 3
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.bn1 = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.action_dim)
        nn.init.constant_(self.fc3.bias, 0.)

    def _setup_training(self):
        self.optimizer = torch.optim.Adam(
            list(self.fc1.parameters()) +
            list(self.fc2.parameters()) +
            list(self.fc3.parameters()), lr=1e-3)

    def select_action(self, state: Dict) -> Tuple[int, torch.Tensor, torch.Tensor]:
        feats = self._build_features(state)
        x = F.gelu(self.bn1(self.fc1(feats)))
        x = self.dropout(x)
        x = F.gelu(self.fc2(x))
        logits = self.fc3(x)
        mask = self._get_action_mask(state)
        
        # 限制logits值范围防止数值不稳定
        logits = torch.clamp(logits, min=-10, max=10)
        
        # UCB探索
        if not hasattr(self, 'action_counts'):
            self.action_counts = torch.zeros_like(logits)
        ucb_scores = logits + self.config.ucb_exploration * torch.sqrt(
            torch.log(torch.sum(self.action_counts)) / (self.action_counts + 1e-6))
        logits = ucb_scores.masked_fill(~mask, float('-inf'))
        
        # 添加数值稳定性检查
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)
        
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        # 更新动作计数
        self.action_counts[0, action] += 1
        self.last_action = int(action.item())
        return int(action.item()), log_prob, entropy


    def parameters(self):
        return list(self.fc1.parameters()) + \
               list(self.fc2.parameters()) + \
               list(self.fc3.parameters())

    def update(self, batch: Dict):
        states: List[Dict] = batch['states']
        actions = torch.tensor(batch['actions'], dtype=torch.int64)
        old_log_probs = torch.stack(batch['log_probs'])
        returns = batch['returns']
        new_log_probs = []
        entropies = []
        for state, action in zip(states, actions):
            _, log_prob, entropy = self.select_action(state)
            new_log_probs.append(log_prob)
            entropies.append(entropy)
        new_log_probs = torch.stack(new_log_probs)
        entropies = torch.stack(entropies)
        
        # 策略损失 + 熵正则化
        policy_loss = -torch.sum(new_log_probs * returns)
        entropy_loss = -torch.sum(entropies) * self.config.entropy_coef
        loss = policy_loss + entropy_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        return loss.item()
