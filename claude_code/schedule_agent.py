import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from typing import Dict, Any, List

from torch_geometric.nn import GATConv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GNNPolicy(nn.Module):
    """图神经网络策略网络，输出每个工序的调度优先级分数（GAT+Dropout）"""
    def __init__(self, node_feat_dim, hidden_dim):
        super().__init__()
        self.conv1 = GATConv(node_feat_dim, hidden_dim, heads=2, concat=True)
        self.conv2 = GATConv(hidden_dim * 2, hidden_dim, heads=2, concat=True)
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.conv2(x, edge_index))
        x = self.dropout(x)
        return self.head(x).squeeze(-1)  # [num_nodes]  
     
class ScheduleAgent:
    """
    基于GNN+规则分配的FJSP调度智能体
    """
    def __init__(self, config):
        self.config = config
        self.node_feat_dim = 10  # 包含优先级等特征
        self.hidden_dim = 32
        self.policy_net = GNNPolicy(self.node_feat_dim, self.hidden_dim)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-3)

    def build_graph(self, state: Dict) -> tuple[Data, list, int]:
        """
        构建异构图：工序节点+机器节点，边为可加工关系
        节点特征包含优先级、剩余工序数、是否等待、是否完成等
        """
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
        # 构建 distributor_id 到 delivery_requirements 的映射
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
        # 1. 添加工序节点特征（归一化/丰富特征）
        for job in jobs:
            if getattr(job, 'status', '') == 'waiting':
                op = job.operations[getattr(job, 'current_operation', 0)]
                # 新增：剩余工序数归一化、剩余到期时间、最短加工时间归一化
                remain_ops = len(job.operations) - getattr(job, 'current_operation', 0)
                weight = getattr(job, 'weight', 1.0)
                min_proc_time = min(op.processing_times.values()) if hasattr(op, 'processing_times') else 0
                # 获取该 job 所属 distributor 的 delivery_requirements
                delivery_req = distributor_req_map.get(job.distributor_id, None)
                # 取最大 due_time 作为最终截止时间（或根据需要取不同阶段的 due_time）
                if delivery_req and hasattr(delivery_req, 'due_times'):
                    due_date = max(delivery_req.due_times)
                else:
                    due_date = 1  # 防止除零

                remain_time_to_due = due_date - state.get('t', 0)
                feat = [
                    getattr(job, 'current_operation', 0) / max(1, len(job.operations)),  # 当前工序归一化
                    remain_ops / max(1, len(job.operations)),                            # 剩余工序归一化
                    1.0 if getattr(job, 'status', '') == 'waiting' else 0.0,
                    1.0 if getattr(job, 'status', '') == 'completed' else 0.0,
                    getattr(job, 'priority', 1.0),
                    due_date / max_due,
                    weight / max_weight,
                    len(op.available_machine_ids) / max(1, len(machines)),  # 可选机器数归一化
                    min_proc_time / max_proc,          # 最短加工时间归一化
                    remain_time_to_due / max_due       # 新增：距离到期时间归一化
                ]
                node_features.append(feat)
                op_node_indices.append((len(node_features)-1, job.job_id, op))
        num_ops = len(op_node_indices)
    
        # 2. 添加机器节点特征（归一化/丰富特征）
        for machine in machines:
            feat = [
                1.0 if machine.status == 'waiting' else 0.0,
                getattr(machine, 'current_job', -1) / max(1, num_jobs),
                getattr(machine, 'remaining_time', 0) / (max_proc if max_proc else 1),
                getattr(machine, 'utilization', 0.0),
                getattr(machine, 'total_processed', 0) / max(1, state.get('t', 1)),  # 新增：历史加工量归一化
                0.0, 0.0, 0.0, 0.0, 0.0
            ]
            node_features.append(feat)
    
        # 3. 构建工序-机器可加工边
        for op_idx, job_id, op in op_node_indices:
            for m in op.available_machine_ids:
                edge_index[0].append(op_idx)
                edge_index[1].append(num_ops + m)  # 机器节点索引
    
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        return Data(x=x, edge_index=edge_index), op_node_indices, num_ops
    
    def select_action(self, state: Dict, return_log_prob: bool = False):
        data, op_node_indices, num_ops = self.build_graph(state)
        if num_ops == 0:
            if return_log_prob:
                return {}, torch.tensor(0.0)
            else:
                return {}

        scores = self.policy_net(data.x, data.edge_index)
        op_scores = scores[:num_ops]

        # 动作掩码
        mask = torch.tensor([
            1 if len(op.available_machine_ids) > 0 else 0
            for _, _, op in op_node_indices
        ], dtype=torch.bool)
        op_scores[~mask] = float('-inf')
        if mask.sum() == 0:
            if return_log_prob:
                return {}, torch.tensor(0.0)
            else:
                return {}

        # 贪心或采样
        if self.config.train_mode == True:
            probs = F.softmax(op_scores, dim=0)
            selected_idx = torch.multinomial(probs, 1).item()
            log_prob = torch.log(probs[int(selected_idx)] + 1e-8)
        else:
            selected_idx = int(torch.argmax(op_scores).item())
            log_prob = torch.tensor(0.0)

        selected_idx = int(selected_idx)
        # 获取对应的作业和工序
        best_job_id = op_node_indices[selected_idx][1]
        best_op = op_node_indices[selected_idx][2]

        # 规则分配最早空闲机器
        machines = state['machines']
        min_time = float('inf')
        best_machine = None
        for m in best_op.available_machine_ids:
            machine = machines[m]
            if machine.status == 'waiting':
                best_machine = m
                break
            elif hasattr(machine, 'remaining_time') and machine.remaining_time < min_time:
                min_time = machine.remaining_time
                best_machine = m
        action = {best_job_id: best_machine} if best_machine is not None else {}

        if return_log_prob:
            return action, log_prob
        else:
            return action
    
    
    def update(self, batch):
        states = batch['states']
        actions = batch['actions']
        log_probs = batch['log_probs']
        returns = batch['returns']
        new_log_probs = []
        for state in states:
            _, log_prob = self.select_action(state, return_log_prob=True)  # 一定要加 return_log_prob=True
            new_log_probs.append(log_prob)
        new_log_probs = torch.stack(new_log_probs)
        loss = -torch.mean(new_log_probs * returns)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()       