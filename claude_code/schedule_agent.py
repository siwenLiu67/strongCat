import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from typing import Dict, Any, List

class GNNPolicy(nn.Module):
    """图神经网络策略网络，输出每个工序的调度优先级分数"""
    def __init__(self, node_feat_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(node_feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)  # 输出每个工序节点的分数

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
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

        # 1. 添加工序节点特征
        for job in jobs:
            # 只考虑可调度（waiting）工序
            if getattr(job, 'status', '') == 'waiting':
                op = job.operations[getattr(job, 'current_operation', 0)]
                feat = [
                    getattr(job, 'current_operation', 0),
                    len(job.operations) - getattr(job, 'current_operation', 0),
                    1.0 if getattr(job, 'status', '') == 'waiting' else 0.0,
                    1.0 if getattr(job, 'status', '') == 'completed' else 0.0,
                    getattr(job, 'priority', 1.0),
                    getattr(job, 'due_date', 0),
                    getattr(job, 'weight', 1.0),
                    getattr(job, 'waiting_time', 0),  # 新增：已等待时间
                    len(op.available_machine_ids),       # 新增：可选机器数
                    min(op.processing_times.values()) if hasattr(op, 'processing_times') else 0  # 新增：最短加工时间
                ]
                node_features.append(feat)
                op_node_indices.append((len(node_features)-1, job.job_id, op))
        num_ops = len(op_node_indices)

        # 2. 添加机器节点特征
        for machine in machines:
            feat = [
                1.0 if machine.status == 'waiting' else 0.0,
                getattr(machine, 'current_job', -1),
                getattr(machine, 'remaining_time', 0),
                getattr(machine, 'utilization', 0.0),
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0
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

    def select_action(self, state: Dict) -> Dict[int, int]:
        """
        选择调度动作：GNN输出每个可调度工序分数，选分数最高的工序，
        并用规则分配其到最早可用机器
        """
        data, op_node_indices, num_ops = self.build_graph(state)
        if num_ops == 0:
            return {}  # 没有可调度工序

        scores = self.policy_net(data.x, data.edge_index)
        # 只取工序节点部分
        op_scores = scores[:num_ops]
        best_op_idx = int(torch.argmax(op_scores).item())
        best_job_id = op_node_indices[best_op_idx][1]
        best_op = op_node_indices[best_op_idx][2]

        # 规则分配：选最早空闲的可用机器
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
        if best_machine is not None:
            return {best_job_id: best_machine}
        else:
            return {}
