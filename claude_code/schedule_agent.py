import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from typing import Dict, Any, List

class GNNPolicy(nn.Module):
    """图神经网络调度策略"""
    def __init__(self, node_feat_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(node_feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, out_dim)  # 输出每个作业-机器分配的分数

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.head(x)

class ScheduleAgent:
    """
    基于图神经网络的调度智能体
    """
    def __init__(self, config):
        self.config = config
        self.node_feat_dim = 8   # 你可以根据实际特征数调整
        self.hidden_dim = 32
        self.policy_net = GNNPolicy(self.node_feat_dim, self.hidden_dim, 1)  # 输出每个作业节点的分数

    def build_graph(self, state: Dict) -> Data:
        """
        根据环境状态构建PyG的Data图对象
        节点包括作业和机器，边可为作业-机器可加工关系
        """
        jobs = state['available_jobs']
        machines = state['machines']
        num_jobs = len(jobs)
        num_machines = len(machines)
        node_features = []
        edge_index = [[], []]

        # 1. 添加作业节点特征
        for job in jobs:
            # 示例特征：[当前工序, 剩余工序数, 是否等待, 是否完成, ...]
            feat = [
                getattr(job, 'current_operation', 0),
                len(job.operations) - getattr(job, 'current_operation', 0),
                1.0 if getattr(job, 'status', '') == 'waiting' else 0.0,
                1.0 if getattr(job, 'status', '') == 'completed' else 0.0,
                0.0, 0.0, 0.0, 0.0  # 预留
            ]
            node_features.append(feat)

        # 2. 添加机器节点特征
        for machine in machines:
            # 示例特征：[是否空闲, 当前作业, ...]
            feat = [
                1.0 if machine.status == 'idle' else 0.0,
                getattr(machine, 'current_job', -1),
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # 预留
            ]
            node_features.append(feat)

        # 3. 构建作业-机器可加工边
        for j, job in enumerate(jobs):
            if hasattr(job, 'operations'):
                op = job.operations[getattr(job, 'current_operation', 0)]
                for m in op.available_machines:
                    edge_index[0].append(j)
                    edge_index[1].append(num_jobs + m)  # 机器节点索引

        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        return Data(x=x, edge_index=edge_index)

    def select_action(self, state: Dict) -> Dict[int, int]:
        """
        基于GNN输出分数，选择调度动作（作业ID -> 机器ID）
        """
        data = self.build_graph(state)
        scores = self.policy_net(data.x, data.edge_index).squeeze(-1)
        num_jobs = len(state['available_jobs'])
        jobs = state['available_jobs']

        # 选择分数最高的作业-机器对（这里只做示例，实际可用mask或采样）
        action = {}
        for j, job in enumerate(jobs):
            # 找到与该作业相连的机器节点
            candidate_machines = []
            for idx, (src, dst) in enumerate(zip(data.edge_index[0], data.edge_index[1])):
                if src == j:
                    candidate_machines.append((dst.item() - num_jobs, scores[dst].item()))
            if candidate_machines:
                # 选分数最高的机器
                best_machine = max(candidate_machines, key=lambda x: x[1])[0]
                action[job.job_id] = best_machine
        return action