import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from typing import Dict, Any, List
from collections import deque

class TransformerPolicy(nn.Module):
    """基于Transformer的调度策略网络"""
    def __init__(self, node_feat_dim, hidden_dim, num_heads=4):
        super().__init__()
        self.node_embedding = nn.Linear(node_feat_dim, hidden_dim)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=0.1
        )
        self.head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, edge_index):
        # 节点嵌入
        x = self.node_embedding(x)
        
        # 使用Transformer处理节点特征
        x = self.transformer(x)
        
        # 输出每个节点的调度优先级分数
        return self.head(x).squeeze(-1)

class ImprovedScheduleAgent:
    """改进的调度智能体，使用PPO算法和Transformer架构"""
    def __init__(self, config):
        self.config = config
        self.node_feat_dim = 10
        self.hidden_dim = 64
        self.policy_net = TransformerPolicy(self.node_feat_dim, self.hidden_dim)
        self.old_policy_net = TransformerPolicy(self.node_feat_dim, self.hidden_dim)
        self.old_policy_net.load_state_dict(self.policy_net.state_dict())
        
        # PPO优化器
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=3e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.95)
        
        # 优先级经验回放
        self.replay_buffer = deque(maxlen=2000)
        self.batch_size = 64
        self.gamma = 0.01
        self.gae_lambda = 0.01
        self.ppo_epochs = 8
        self.clip_param = 0.15
        
    def build_graph(self, state: Dict) -> tuple[Data, list, int]:
        """构建调度图结构（从原schedule_agent.py复制完整实现）"""
        jobs = state['available_jobs']
        machines = state['machines']
        num_jobs = len(jobs)
        node_features = []
        edge_index = [[], []]
        op_node_indices = []  # 记录每个可调度工序的节点索引及其作业id
    
        # 统计全局最大最小值用于归一化
        all_weights = [getattr(job, 'weight', 1.0) for job in jobs]
        all_proc_times = []
        for job in jobs:
            op = job.operations[getattr(job, 'current_operation', 0)]
            if hasattr(op, 'processing_times'):
                all_proc_times += list(op.processing_times.values())
        distributors = state.get('distributors', [])
        distributor_req_map = {
            d.distributor_id: d.delivery_requirements for d in distributors
        }
        max_weight = max(all_weights) if all_weights else 1
        max_proc = max(all_proc_times) if all_proc_times else 1
        all_due_times = []
        for d in distributors:
            if hasattr(d, 'delivery_requirements') and hasattr(d.delivery_requirements, 'due_times'):
                all_due_times += list(d.delivery_requirements.due_times)
        max_due = max(all_due_times) if all_due_times and max(all_due_times) > 0 else 1

        # 1. 添加工序节点特征
        for job in jobs:
            if getattr(job, 'status', '') == 'waiting':
                op = job.operations[getattr(job, 'current_operation', 0)]
                remain_ops = len(job.operations) - getattr(job, 'current_operation', 0)
                weight = getattr(job, 'weight', 1.0)
                min_proc_time = min(op.processing_times.values()) if hasattr(op, 'processing_times') else 0
                delivery_req = distributor_req_map.get(job.distributor_id, None)
                if delivery_req and hasattr(delivery_req, 'due_times'):
                    due_date = max(delivery_req.due_times)
                else:
                    due_date = 1

                remain_time_to_due = due_date - state.get('t', 0)
                feat = [
                    getattr(job, 'current_operation', 0) / max(1, len(job.operations)),
                    remain_ops / max(1, len(job.operations)),
                    1.0 if getattr(job, 'status', '') == 'waiting' else 0.0,
                    1.0 if getattr(job, 'status', '') == 'completed' else 0.0,
                    getattr(job, 'priority', 1.0),
                    due_date / max_due,
                    weight / max_weight,
                    len(op.available_machine_ids) / max(1, len(machines)),
                    min_proc_time / max_proc,
                    remain_time_to_due / max_due
                ]
                node_features.append(feat)
                op_node_indices.append((len(node_features)-1, job.job_id, op))
        num_ops = len(op_node_indices)
    
        # 2. 添加机器节点特征
        for machine in machines:
            feat = [
                1.0 if machine.status == 'waiting' else 0.0,
                getattr(machine, 'current_job', -1) / max(1, num_jobs),
                getattr(machine, 'remaining_time', 0) / (max_proc if max_proc else 1),
                getattr(machine, 'utilization', 0.0),
                getattr(machine, 'total_processed', 0) / max(1, state.get('t', 1)),
                0.0, 0.0, 0.0, 0.0, 0.0
            ]
            node_features.append(feat)
    
        # 3. 构建工序-机器可加工边
        for op_idx, job_id, op in op_node_indices:
            for m in op.available_machine_ids:
                edge_index[0].append(op_idx)
                edge_index[1].append(num_ops + m)

        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        return Data(x=x, edge_index=edge_index), op_node_indices, num_ops
        
    def select_action(self, state: Dict, return_log_prob: bool = False):
        """选择动作（基于Transformer策略）"""
        data, op_node_indices, num_ops = self.build_graph(state)
        if num_ops == 0:
            if return_log_prob:
                return {}, torch.tensor(0.0)
            return {}
            
        scores = self.policy_net(data.x, data.edge_index)
        op_scores = scores[:num_ops]  # 使用切片语法
        
        # 动作掩码
        mask = torch.tensor([
            1 if len(op.available_machine_ids) > 0 else 0
            for _, _, op in op_node_indices
        ], dtype=torch.bool)
        op_scores[~mask] = float('-inf')
        
        if mask.sum() == 0:
            if return_log_prob:
                return {}, torch.tensor(0.0)
            return {}
            
        # 动作选择
        probs = F.softmax(op_scores, dim=0)
        dist = torch.distributions.Categorical(probs)
        selected_idx = dist.sample()
        log_prob = dist.log_prob(selected_idx)
        
        # 机器选择规则
        idx = int(selected_idx.item())
        best_job_id = op_node_indices[idx][1]
        best_op = op_node_indices[idx][2]
        best_machine = self._select_machine(state['machines'], best_op)
        
        action = {best_job_id: best_machine} if best_machine is not None else {}
        
        if return_log_prob:
            return action, log_prob
        return action
        
    def _select_machine(self, machines, op):
        """机器选择策略"""
        # 可以扩展为更智能的机器选择
        for m in op.available_machine_ids:
            if machines[m].status == 'waiting':
                return m
        return random.choice(op.available_machine_ids)
        
    def store_experience(self, batch):
        """存储经验到回放缓冲区"""
        self.replay_buffer.append(batch)

    def update(self, batch_data):
        """PPO更新策略（批量处理，动作索引与log_prob严格对应）"""
        print("Updating policy with batch size:", len(batch_data['states']))
        print("Replay buffer size:", len(self.replay_buffer))
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        losses = []
        for _ in range(self.ppo_epochs):
            # 采集所有 step 的数据
            states = batch_data['states']
            old_log_probs = torch.tensor(batch_data['log_probs'], dtype=torch.float32)
            returns = torch.tensor(batch_data['returns'], dtype=torch.float32)
            values = returns  # 如果没有单独的 value
            selected_idx_list = batch_data['selected_idx_list']  # 新增：采集时保存的动作索引

            # 构建批量图
            data_list = []
            op_indices = []
            for state in states:
                data, op_node_indices, num_ops = self.build_graph(state)
                data_list.append(data)
                op_indices.append(op_node_indices)
            from torch_geometric.data import Batch
            batch_graph = Batch.from_data_list(data_list)
            scores = self.policy_net(batch_graph.x, batch_graph.edge_index)

            # 计算 new_log_probs（严格用采集时的动作索引）
            new_log_probs = []
            idx = 0
            for i, op_node_indices in enumerate(op_indices):
                num_ops = len(op_node_indices)
                op_scores = scores[idx:idx+num_ops]
                probs = F.softmax(op_scores, dim=0)
                dist = torch.distributions.Categorical(probs)
                # 用采集时的动作索引 selected_idx_list[i]
                log_prob = dist.log_prob(torch.tensor(selected_idx_list[i]))
                new_log_probs.append(log_prob)
                idx += num_ops
            new_log_probs = torch.stack(new_log_probs)

    
            # PPO损失
            advantages = returns - values
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(returns, values)
            entropy_loss = -torch.mean(new_log_probs)
            loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.optimizer.step()
            losses.append(loss.item())

        self.scheduler.step()
        self.old_policy_net.load_state_dict(self.policy_net.state_dict())

        return np.mean(losses)