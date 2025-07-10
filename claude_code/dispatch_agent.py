import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from typing import Dict, Any, List

class GNNDispatchPolicy(nn.Module):
    """图神经网络配送策略网络，输出每个可配送作业的分批优先级分数"""
    def __init__(self, node_feat_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(node_feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)  # 输出每个作业节点的分数

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.head(x).squeeze(-1)  # [num_nodes]

class DispatchAgent:
    """
    基于GNN+规则分批的配送智能体
    """
    def __init__(self, config):
        self.config = config
        self.node_feat_dim = 8  # 可根据实际特征调整
        self.hidden_dim = 32
        self.policy_net = GNNDispatchPolicy(self.node_feat_dim, self.hidden_dim)

    def build_graph(self, state: Dict) -> tuple[Data, list]:
        """
        构建作业节点图，节点特征包含优先级、交期、权重、等待时间等
        """
        jobs = state['completed_jobs']
        node_features = []
        job_indices = []

        for job in jobs:
            # 只考虑已完成、待配送的作业
            if getattr(job, 'status', '') == 'completed':
                feat = [
                    getattr(job, 'priority', 1.0),
                    getattr(job, 'due_date', 0),
                    getattr(job, 'weight', 1.0),
                    getattr(job, 'waiting_time', 0),
                    getattr(job, 'delivery_time', 0) if hasattr(job, 'delivery_time') else 0,
                    0.0, 0.0, 0.0  # 预留
                ]
                node_features.append(feat)
                job_indices.append(job.job_id)

        x = torch.tensor(node_features, dtype=torch.float) if node_features else torch.zeros((1, self.node_feat_dim))
        edge_index = torch.zeros((2, 0), dtype=torch.long)  # 无边
        return Data(x=x, edge_index=edge_index), job_indices

    def select_action(self, state: Dict) -> Dict[int, List[int]]:
        """
        选择配送动作：GNN输出每个可配送作业分数，按配送商可配送范围分批
        返回格式：{distributor_id: [job_id1, job_id2, ...]}
        """
        data, job_indices = self.build_graph(state)
        if len(job_indices) == 0:
            return {}

        scores = self.policy_net(data.x, data.edge_index)
        sorted_idx = torch.argsort(scores, descending=True)
        sorted_jobs = [job_indices[i] for i in sorted_idx.tolist()]

        distributors = state.get('distributors', [])
        if not distributors:
            return {}

        # 构建配送商可配送作业映射
        distributor_batches = {}
        for distributor in distributors:
            # 假设每个配送商有 can_deliver_job_ids 属性
            can_deliver = set(getattr(distributor, 'assigned_jobs', []))
            batch = [job_id for job_id in sorted_jobs if job_id in can_deliver]
            if batch:
                distributor_id = getattr(distributor, 'distributor_id', 0)
                distributor_batches[distributor_id] = batch

        return distributor_batches