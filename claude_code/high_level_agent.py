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
        # 初始化持久化的动作计数
        self.action_counts = torch.ones(1, 3)  # 初始为1，避免除零
        self.total_steps = 0

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
        
        # 修复：检查是否有未配送的已完成作业
        completed_job_ids = {getattr(j, 'job_id', i) for i, j in enumerate(completed_jobs)}
        dispatched_job_ids = {getattr(j, 'job_id', i) for i, j in enumerate(dispatched_jobs)}
        
        # 如果没有未配送的已完成作业，禁用配送动作
        if not completed_job_ids or len(completed_job_ids - dispatched_job_ids) == 0:
            mask[1] = False

        # 如果都在配送中状态，禁用配送动s作
        if all(getattr(j, 'status') in ['dispatching', 'dispatched'] for j in state['available_jobs']):
            mask[1] = False
       
        # 如果没有可调度作业，禁用调度动作
        if not state['available_jobs'] or not(getattr(j, 'status') in ['waiting'] for j in state['available_jobs']):
            mask[0] = False
        
        # 只要有空闲机器且有可调度作业，就允许schedule
        if not state['machines'] or all(m.remaining_time > 0 for m in state['machines']):
            mask[0] = False
            
        # 打印调试信息
        print(f"Action mask: {mask}, Completed jobs: {len(completed_jobs)}, Dispatched jobs: {len(dispatched_jobs)}")
        
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
        
        # 获取动作掩码并打印
        mask = self._get_action_mask(state)
        print(f"Raw logits before masking: {logits}")
        
        # 限制logits值范围防止数值不稳定
        logits = torch.clamp(logits, min=-10, max=10)
        
        # 统一的 UCB 探索逻辑
        self.total_steps += 1
        
        # 打印原始动作概率和掩码情况
        print(f"Action mask: {mask.squeeze()}")
        print(f"Available actions: {[i for i, m in enumerate(mask.squeeze()) if m]}")
        
        # 应用UCB探索
        exploration_bonus = self.config.ucb_exploration * torch.sqrt(
            torch.log(torch.tensor(self.total_steps)) / (self.action_counts + 1e-6))
        
        ucb_scores = logits + exploration_bonus
        print(f"UCB scores before masking: {ucb_scores}")
        
        # 应用动作掩码（只应用一次）
        masked_scores = ucb_scores.masked_fill(~mask, float('-inf'))
        
        # 添加数值稳定性检查
        if torch.isnan(masked_scores).any() or torch.isinf(masked_scores).any():
            masked_scores = torch.nan_to_num(masked_scores, nan=0.0, posinf=1.0, neginf=-1.0)
        masked_scores = ucb_scores.masked_fill(~mask, float('-inf'))

        # 打印最终分数
        print(f"Final action scores after masking: {masked_scores}")
        
        probs = F.softmax(masked_scores, dim=-1)
        probs = probs * mask  # 确保无效动作的概率为 0
        probs = probs / probs.sum(dim=-1, keepdim=True)  # 重新
        print(f"Action probabilities: {probs}")
        
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        # 更新动作计数
        self.action_counts[0, action] += 1
        
        # 打印选择的动作
        self.last_action = int(action.item())
        print(f"Selected action: {self.last_action}")
        
        return self.last_action, log_prob, entropy


    def parameters(self):
        return list(self.fc1.parameters()) + \
               list(self.fc2.parameters()) + \
               list(self.fc3.parameters())

    def update(self, batch: Dict, optimizer, scheduler) -> float:
        """更新高层策略网络（批量处理）"""
        states: List[Dict] = batch['states']
        actions = torch.tensor(batch['actions'], dtype=torch.int64)
        old_log_probs = torch.stack(batch['log_probs'])
        returns = batch['returns']

        # 批量构建特征
        features = torch.cat([self._build_features(state) for state in states], dim=0)
        
        # 批量前向传播
        x = F.gelu(self.bn1(self.fc1(features)))
        x = self.dropout(x)
        x = F.gelu(self.fc2(x))
        logits = self.fc3(x)
        
        # 计算原始动作的新概率
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        new_log_probs = dist.log_prob(actions)
        entropies = dist.entropy()
        
        # 策略损失 + 熵正则化
        policy_loss = -torch.mean(new_log_probs * returns)  # 均值更稳定
        entropy_loss = -torch.mean(entropies) * self.config.entropy_coef
        loss = policy_loss + entropy_loss
        
        # 打印调试信息
        print(f"Policy loss: {policy_loss.item():.4f}, Entropy: {entropy_loss.item():.4f}")
        
        # 关键修正：使用传入的optimizer，而不是self.optimizer
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 关键修正：更新学习率调度器
        scheduler.step()
        
        return loss.item()