"""
使用分层强化学习(HRL) + 图注意力网络(GAT)解决集成柔性作业车间调度与派遣问题(IFJSSP-DP)
2层图结构，注意力机制，离散动作空间
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


class GraphAttentionLayer(nn.Module):
    """图注意力层"""
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1, alpha: float = 0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(self, h, adj):
        """
        h: [batch_size, num_nodes, in_features]
        adj: [batch_size, num_nodes, num_nodes]
        """
        batch_size, num_nodes = h.size(0), h.size(1)
        
        # 线性变换
        Wh = torch.matmul(h, self.W)  # [batch_size, num_nodes, out_features]
        
        # 计算注意力系数
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        
        # 掩码处理（仅计算连接的节点）
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # 应用注意力权重
        h_prime = torch.matmul(attention, Wh)
        
        return h_prime
    
    def _prepare_attentional_mechanism_input(self, Wh):
        """准备注意力机制输入"""
        batch_size, num_nodes = Wh.size(0), Wh.size(1)
        
        # 扩展维度以便计算所有节点对的注意力
        Wh_repeated_in_chunks = Wh.repeat_interleave(num_nodes, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, num_nodes, 1)
        
        # 拼接特征
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)
        
        return all_combinations_matrix.view(batch_size, num_nodes, num_nodes, 2 * self.out_features)


class MultiHeadGATLayer(nn.Module):
    """多头图注意力层"""
    
    def __init__(self, in_features: int, out_features: int, num_heads: int = 8, dropout: float = 0.1):
        super(MultiHeadGATLayer, self).__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(in_features, out_features, dropout)
            for _ in range(num_heads)
        ])
        
        self.out_proj = nn.Linear(num_heads * out_features, out_features)
        
    def forward(self, h, adj):
        """多头注意力前向传播"""
        head_outputs = []
        for attention in self.attentions:
            head_outputs.append(attention(h, adj))
        
        # 拼接所有头的输出
        multi_head_output = torch.cat(head_outputs, dim=2)
        
        # 投影到最终输出维度
        output = self.out_proj(multi_head_output)
        
        return output


class HierarchicalGATNetwork(nn.Module):
    """分层图注意力网络"""
    
    def __init__(self, node_features: int, hidden_dim: int = 128, num_heads: int = 4):
        super(HierarchicalGATNetwork, self).__init__()
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        
        # 第一层：任务级图注意力（作业-机器关系）
        self.task_gat_layer1 = MultiHeadGATLayer(node_features, hidden_dim, num_heads)
        self.task_gat_layer2 = MultiHeadGATLayer(hidden_dim, hidden_dim, num_heads)
        
        # 第二层：资源级图注意力（机器-配送商关系）  
        self.resource_gat_layer1 = MultiHeadGATLayer(hidden_dim, hidden_dim, num_heads)
        self.resource_gat_layer2 = MultiHeadGATLayer(hidden_dim, hidden_dim, num_heads)
        
        # 高层策略网络
        self.high_level_policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # 4个高层动作：调度、派遣、等待、优化
        )
        
        # 低层策略网络
        self.low_level_policies = nn.ModuleDict({
            'schedule': nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)  # 调度优先级
            ),
            'dispatch': nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)  # 派遣优先级
            ),
            'wait': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)  # 等待时间
            ),
            'optimize': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)  # 优化参数
            )
        })
        
        # 价值函数
        self.value_function = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, node_features, task_adj, resource_adj):
        """
        node_features: [batch_size, num_nodes, node_features]
        task_adj: [batch_size, num_nodes, num_nodes] 任务级邻接矩阵
        resource_adj: [batch_size, num_nodes, num_nodes] 资源级邻接矩阵
        """
        # 第一层：任务级图注意力
        task_h1 = F.relu(self.task_gat_layer1(node_features, task_adj))
        task_h2 = F.relu(self.task_gat_layer2(task_h1, task_adj))
        
        # 第二层：资源级图注意力
        resource_h1 = F.relu(self.resource_gat_layer1(task_h2, resource_adj))
        resource_h2 = F.relu(self.resource_gat_layer2(resource_h1, resource_adj))
        
        # 全局特征聚合
        global_features = torch.mean(resource_h2, dim=1)  # [batch_size, hidden_dim]
        
        # 高层策略
        high_level_action_probs = F.softmax(self.high_level_policy(global_features), dim=-1)
        
        # 状态价值
        state_value = self.value_function(global_features)
        
        return {
            'node_embeddings': resource_h2,
            'global_features': global_features,
            'high_level_probs': high_level_action_probs,
            'state_value': state_value
        }
    
    def get_low_level_action(self, node_embeddings, global_features, high_level_action, node_pairs):
        """获取低层动作"""
        action_type = ['schedule', 'dispatch', 'wait', 'optimize'][high_level_action]
        
        if action_type in ['schedule', 'dispatch'] and node_pairs is not None:
            # 对于调度和派遣，计算节点对的优先级
            node1_emb = node_embeddings[:, node_pairs[:, 0]]  # [batch_size, num_pairs, hidden_dim]
            node2_emb = node_embeddings[:, node_pairs[:, 1]]  # [batch_size, num_pairs, hidden_dim]
            pair_features = torch.cat([node1_emb, node2_emb], dim=-1)
            
            priorities = self.low_level_policies[action_type](pair_features).squeeze(-1)
            return F.softmax(priorities, dim=-1)
        else:
            # 对于等待和优化动作
            if action_type == 'wait':
                return self.low_level_policies[action_type](global_features)
            else:
                return self.low_level_policies[action_type](global_features)


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, graph_state, high_action, low_action, reward, next_graph_state, done):
        self.buffer.append((graph_state, high_action, low_action, reward, next_graph_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return zip(*batch)
    
    def __len__(self):
        return len(self.buffer)


class FJSSPGraphEnvironment:
    """FJSSP-DP图环境"""
    
    def __init__(self, scenario: FlexibleJobShopScenario):
        self.scenario = scenario
        self.jobs = scenario.jobs
        self.machines = scenario.machines
        self.distributors = scenario.distributors
        
        # 图结构参数
        self.num_job_nodes = len(self.jobs)
        self.num_machine_nodes = len(self.machines)
        self.num_distributor_nodes = len(self.distributors)
        self.total_nodes = self.num_job_nodes + self.num_machine_nodes + self.num_distributor_nodes
        
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
            machine.status = "waiting"
            machine.current_job = -1
            machine.remaining_time = 0.0
            machine.processed_jobs = []
            machine.total_busy_time = 0.0
            machine.total_idle_time = 0.0
        
        # 重置作业状态
        for job in self.jobs:
            job.status = "waiting"
            job.current_operation = 0
            job.completed_time = 0.0
            job.dispatched_time = 0.0
             
        self._update_available_jobs()
        return self._get_graph_state()
    
    def _update_available_jobs(self):
        """更新可用作业列表"""
        for job in self.pending_jobs[:]:
            arrival_time = getattr(job, 'arrival_time', 0)
            if arrival_time <= self.current_time:
                self.available_jobs.append(job)
                self.pending_jobs.remove(job)
    
    def _get_graph_state(self) -> Dict:
        """获取图状态表示"""
        # 节点特征矩阵
        node_features = np.zeros((self.total_nodes, 10), dtype=np.float32)
        
        # 作业节点特征
        for i, job in enumerate(self.jobs):
            features = [
                1.0,  # 节点类型：作业
                job.current_operation / max(len(job.operations), 1),  # 进度
                1.0 if job.status == "waiting" else 0.0,  # 等待状态
                1.0 if job.status == "processing" else 0.0,  # 处理状态
                1.0 if job.status == "completed" else 0.0,  # 完成状态
                getattr(job, 'due_date', 100) / 100.0,  # 截止时间
                max(0, getattr(job, 'due_date', 100) - self.current_time) / 100.0,  # 紧急度
                1.0,  # 默认优先级
                len(job.operations) / 10.0,  # 工序数量
                self.current_time / 100.0  # 当前时间
            ]
            node_features[i] = features
        
        # 机器节点特征
        for i, machine in enumerate(self.machines):
            idx = self.num_job_nodes + i
            features = [
                2.0,  # 节点类型：机器
                1.0 if machine.status == "waiting" else 0.0,  # 空闲状态
                1.0 if machine.status == "busy" else 0.0,  # 忙碌状态
                machine.remaining_time / 20.0,  # 剩余处理时间
                len(machine.processed_jobs) / max(len(self.jobs), 1),  # 已处理作业比例
                machine.total_busy_time / max(self.current_time, 1),  # 利用率
                0.5,  # 默认能力指标
                1.0,  # 默认效率
                0.0,  # 预留
                self.current_time / 100.0  # 当前时间
            ]
            node_features[idx] = features
        
        # 配送商节点特征
        for i, distributor in enumerate(self.distributors):
            idx = self.num_job_nodes + self.num_machine_nodes + i
            dist_jobs = [j for j in self.jobs if getattr(j, 'distributor_id', i % len(self.distributors)) == distributor.distributor_id]
            completed_for_dist = [j for j in dist_jobs if j in self.completed_jobs]
            dispatched_for_dist = [j for j in dist_jobs if j in self.dispatched_jobs]
            
            features = [
                3.0,  # 节点类型：配送商
                len(completed_for_dist) / max(len(dist_jobs), 1),  # 完成比例
                len(dispatched_for_dist) / max(len(dist_jobs), 1),  # 派遣比例
                getattr(distributor, 'capacity', 10) / 20.0,  # 容量
                len(dist_jobs) / max(len(self.jobs), 1),  # 负责作业比例
                0.0,  # 当前负载（可扩展）
                0.0,  # 预留
                0.0,  # 预留
                0.0,  # 预留
                self.current_time / 100.0  # 当前时间
            ]
            node_features[idx] = features
        
        # 构建邻接矩阵
        task_adj = self._build_task_adjacency_matrix()
        resource_adj = self._build_resource_adjacency_matrix()
        
        return {
            'node_features': torch.FloatTensor(node_features).unsqueeze(0),
            'task_adj': torch.FloatTensor(task_adj).unsqueeze(0),
            'resource_adj': torch.FloatTensor(resource_adj).unsqueeze(0)
        }
    
    def _build_task_adjacency_matrix(self) -> np.ndarray:
        """构建任务级邻接矩阵（作业-机器关系）"""
        adj = np.zeros((self.total_nodes, self.total_nodes), dtype=np.float32)
        
        # 作业到机器的连接
        for i, job in enumerate(self.jobs):
            if job.current_operation < len(job.operations):
                current_op = job.operations[job.current_operation]
                for machine_id in current_op.available_machine_ids:
                    if machine_id < len(self.machines):
                        machine_idx = self.num_job_nodes + machine_id
                        adj[i][machine_idx] = 1.0
                        adj[machine_idx][i] = 1.0  # 双向连接
        
        # 作业间的前驱关系
        for i, job1 in enumerate(self.jobs):
            for j, job2 in enumerate(self.jobs):
                if i != j:
                    # 基于紧急度的软连接
                    urgency1 = max(0, getattr(job1, 'due_date', 100) - self.current_time)
                    urgency2 = max(0, getattr(job2, 'due_date', 100) - self.current_time)
                    if abs(urgency1 - urgency2) < 20:  # 相似紧急度
                        adj[i][j] = 0.3
        
        return adj
    
    def _build_resource_adjacency_matrix(self) -> np.ndarray:
        """构建资源级邻接矩阵（机器-配送商关系）"""
        adj = np.zeros((self.total_nodes, self.total_nodes), dtype=np.float32)
        
        # 机器间的连接（基于机器ID相似性）
        for i, machine1 in enumerate(self.machines):
            for j, machine2 in enumerate(self.machines):
                if i != j:
                    idx1 = self.num_job_nodes + i
                    idx2 = self.num_job_nodes + j
                    
                    # 基于机器ID的连接强度
                    machine_id_1 = getattr(machine1, 'machine_id', i)
                    machine_id_2 = getattr(machine2, 'machine_id', j)
                    similarity = 1.0 / (1.0 + abs(machine_id_1 - machine_id_2))
                    adj[idx1][idx2] = similarity * 0.5  # 适度连接
        
        # 机器到配送商的连接
        for i, machine in enumerate(self.machines):
            machine_idx = self.num_job_nodes + i
            for j, distributor in enumerate(self.distributors):
                dist_idx = self.num_job_nodes + self.num_machine_nodes + j
                
                # 基于地理位置的连接（简化处理）
                machine_location = getattr(machine, 'location', (i * 10, 0))
                dist_location = getattr(distributor, 'location', (100 + j * 50, 50))
                
                # 计算简化的距离权重
                try:
                    distance = ((machine_location[0] - dist_location[0])**2 + 
                               (machine_location[1] - dist_location[1])**2)**0.5
                    distance_weight = 1.0 / (1.0 + distance / 100.0)  # 标准化距离
                except:
                    distance_weight = 0.5  # 默认权重
                
                adj[machine_idx][dist_idx] = distance_weight
                adj[dist_idx][machine_idx] = distance_weight
        
        # 配送商间的连接
        for i, dist1 in enumerate(self.distributors):
            for j, dist2 in enumerate(self.distributors):
                if i != j:
                    idx1 = self.num_job_nodes + self.num_machine_nodes + i
                    idx2 = self.num_job_nodes + self.num_machine_nodes + j
                    adj[idx1][idx2] = 0.2  # 配送商间的协作关系
        
        return adj
    
    def _get_valid_node_pairs(self, high_action: int) -> List[Tuple[int, int]]:
        """获取有效的节点对"""
        valid_pairs = []
        
        if high_action == 0:  # 调度动作
            for i, job in enumerate(self.available_jobs):
                if job.status == "waiting" and job.current_operation < len(job.operations):
                    current_op = job.operations[job.current_operation]
                    for machine_id in current_op.available_machine_ids:
                        if machine_id < len(self.machines):
                            machine = self.machines[machine_id]
                            if machine.status == "waiting":
                                machine_idx = self.num_job_nodes + machine_id
                                valid_pairs.append((job.job_id, machine_idx))
        
        elif high_action == 1:  # 派遣动作
            completed_waiting = [j for j in self.completed_jobs if j not in self.dispatched_jobs]
            for job in completed_waiting:
                dist_idx = self.num_job_nodes + self.num_machine_nodes + getattr(job, 'distributor_id', 0)
                valid_pairs.append((job.job_id, dist_idx))
        
        return valid_pairs
    
    def step(self, high_action: int, low_action_params: torch.Tensor) -> Tuple[Dict, float, bool, dict]:
        """执行分层动作"""
        reward = 0
        info = {'high_action': high_action}
        
        if high_action == 0:  # 调度
            reward += self._execute_schedule_action(low_action_params)
        elif high_action == 1:  # 派遣
            reward += self._execute_dispatch_action(low_action_params)
        elif high_action == 2:  # 等待
            reward += self._execute_wait_action(low_action_params)
        elif high_action == 3:  # 优化
            reward += self._execute_optimize_action(low_action_params)
        
        # 推进时间
        self._advance_time()
        self._update_available_jobs()
        
        # 计算延误惩罚
        tardiness_penalty = self._calculate_tardiness_penalty()
        reward -= tardiness_penalty
        
        # 检查完成条件
        done = self._is_done()
        
        next_graph_state = self._get_graph_state()
        
        return next_graph_state, reward, done, info
    
    # ...existing code...

    def _execute_schedule_action(self, action_params: torch.Tensor) -> float:
        """执行调度动作"""
        valid_pairs = self._get_valid_node_pairs(0)
        if not valid_pairs:
            return -0.1
        
        # 选择优先级最高的作业-机器对
        if len(action_params.shape) > 0 and len(action_params) >= len(valid_pairs):
            priorities = action_params[:len(valid_pairs)]
            best_idx = torch.argmax(priorities).item()
        else:
            best_idx = 0
        
        # 确保索引在有效范围内
        best_idx = max(0, min(best_idx, len(valid_pairs) - 1))
        
        if best_idx >= len(valid_pairs):
            return -0.1
            
        job_id, machine_idx = valid_pairs[int(best_idx)]
        machine_id = machine_idx - self.num_job_nodes

        result = self._schedule_job(job_id, machine_id)
        if result is None:
            return -1.0
        return float(result)

    def _execute_dispatch_action(self, action_params: torch.Tensor) -> float:
        """执行派遣动作"""
        valid_pairs = self._get_valid_node_pairs(1)
        if not valid_pairs:
            return -0.1
        
        # 按配送商分组派遣
        total_reward = 0
        
        for dist_id in range(len(self.distributors)):
            dist_jobs = [pair[0] for pair in valid_pairs 
                        if pair[1] == self.num_job_nodes + self.num_machine_nodes + dist_id]
            if dist_jobs:
                total_reward += self._dispatch_jobs(dist_id)
        
        return total_reward
    
    
    def _execute_wait_action(self, action_params: torch.Tensor) -> float:
        """执行等待动作"""
        # 等待一定时间，给予小的负奖励
        if len(action_params.shape) > 0 and action_params.numel() > 0:
            wait_time = max(1, int(action_params.item() * 5))
        else:
            wait_time = 1
        return -0.05 * wait_time
        
    def _execute_optimize_action(self, action_params: torch.Tensor) -> float:
        """执行优化动作"""
        # 优化当前调度，重新排列等待队列
        if self.available_jobs:
            # 按紧急度重新排序
            self.available_jobs.sort(key=lambda j: getattr(j, 'due_date', 100) - self.current_time)
            return 0.1
        return 0
    
    def _schedule_job(self, job_id: int, machine_id: int) -> float:
        """调度作业到机器"""
        job = next((j for j in self.available_jobs if j.job_id == job_id), None)
        if not job or job.status != "waiting":
            return -1
        
        if machine_id >= len(self.machines):
            return -1
            
        machine = self.machines[machine_id]
        if machine.status != "waiting":
            return -1
        
        if job.current_operation >= len(job.operations):
            return -1
            
        current_op = job.operations[job.current_operation]
        if machine_id not in current_op.available_machine_ids:
            return -1
        
        processing_time = current_op.processing_times.get(machine_id, 0)
        if processing_time <= 0:
            return -1
            
        machine.assign_job(job_id, processing_time)
        machine.status = "busy"
        
        job.status = "processing"
        if not hasattr(job, 'start_time') or job.start_time is None:
            job.start_time = self.current_time
        
        # 分层奖励：基础奖励 + 图结构奖励
        due_date = getattr(job, 'due_date', 100)
        urgency = max(0, due_date - self.current_time) / max(due_date, 1)
        base_reward = 2.0 + urgency
        
        # 图结构奖励：考虑机器利用率平衡
        machine_loads = []
        for m in self.machines:
            load = len([j for j in self.jobs if getattr(j, 'assigned_machine', -1) == getattr(m, 'machine_id', -1)])
            machine_loads.append(load)
        
        load_variance = np.var(machine_loads) if machine_loads else 0
        balance_reward = 0.5 if load_variance < 2.0 else 0.0
        
        return base_reward + balance_reward
    
    def _dispatch_jobs(self, distributor_id: int) -> float:
        """派遣作业"""
        completed_waiting = [j for j in self.completed_jobs 
                           if j not in self.dispatched_jobs 
                           and getattr(j, 'distributor_id', 0) == distributor_id]
        
        if not completed_waiting:
            return -0.5
        
        reward = 0
        for job in completed_waiting:
            job.dispatch_time = self.current_time
            job.dispatched_time = self.current_time
            job.status = "dispatched"
            self.dispatched_jobs.append(job)
            
            due_date = getattr(job, 'due_date', 100)
            if job.completed_time <= due_date:
                reward += 1.0
            else:
                reward += 0.5
        
        # 批次奖励
        batch_size = len(completed_waiting)
        reward += batch_size * 0.3
        
        return reward
    
    def _advance_time(self):
        """推进时间"""
        # 强制推进时间，避免死循环
        time_advanced = False
        
        # 检查是否有机器在工作
        for machine in self.machines:
            if machine.status == "busy":
                machine.remaining_time -= 1
                machine.total_busy_time += 1
                time_advanced = True
                
                if machine.remaining_time <= 0:
                    job_id = machine.current_job
                    job = next((j for j in self.jobs if j.job_id == job_id), None)
                    
                    if job:
                        job.current_operation += 1
                        if job.current_operation >= len(job.operations):
                            job.status = "completed"
                            job.completed_time = self.current_time + 1
                            if job not in self.completed_jobs:
                                self.completed_jobs.append(job)
                        else:
                            job.status = "waiting"
                            # 将作业重新加入可用列表
                            if job not in self.available_jobs:
                                self.available_jobs.append(job)
                    
                    machine.status = "waiting"
                    machine.current_job = -1
                    machine.remaining_time = 0.0
            else:
                machine.total_idle_time += 1
        
        # 无论如何都要推进时间，避免死循环
        self.current_time += 1
        
        # 如果没有机器在工作且没有可用作业，强制完成一些作业
        if not time_advanced and not self.available_jobs:
            for job in self.jobs:
                if job.status == "waiting" and job not in self.completed_jobs:
                    job.status = "completed"
                    job.completed_time = self.current_time
                    if job not in self.completed_jobs:
                        self.completed_jobs.append(job)
                    break

    
    def _calculate_tardiness_penalty(self) -> float:
        """计算延误惩罚"""
        penalty = 0
        for job in self.completed_jobs:
            due_date = getattr(job, 'due_date', 100)
            if job.completed_time > due_date:
                penalty += (job.completed_time - due_date) * 0.1
        return penalty
    
    def _is_done(self) -> bool:
        """检查是否完成"""
        all_jobs_completed = len(self.completed_jobs) == len(self.jobs)
        all_jobs_dispatched = len(self.dispatched_jobs) == len(self.jobs)
        timeout = self.current_time > 200  # 减少超时时间
        
        # 添加更多完成条件
        no_pending_jobs = len(self.pending_jobs) == 0
        no_available_jobs = len(self.available_jobs) == 0
        all_machines_idle = all(m.status == "waiting" for m in self.machines)
        
        # 如果所有机器都空闲且没有可用作业，提前结束
        early_termination = no_available_jobs and all_machines_idle and no_pending_jobs
        
        return (all_jobs_completed and all_jobs_dispatched) or timeout or early_termination

class HRLGATAgent:
    """分层强化学习 + 图注意力网络智能体"""
    
    def __init__(self, node_features: int, hidden_dim: int = 128, config=None):
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.config = config
        
        # 网络参数
        self.lr = 0.001
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 10000
        self.target_update_freq = 100
        self.batch_size = 16  # 图网络通常用较小的batch size
        self.memory_size = 5000
        
        # 分层GAT网络
        self.policy_net = HierarchicalGATNetwork(node_features, hidden_dim)
        self.target_net = HierarchicalGATNetwork(node_features, hidden_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        # 经验回放
        self.memory = ReplayBuffer(self.memory_size)
        
        # 训练计数器
        self.steps = 0
        self.epsilon = self.epsilon_start
        
        # 统计信息
        self.high_action_usage = defaultdict(int)
        
        # 更新目标网络
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def select_action(self, graph_state: Dict) -> Tuple[int, torch.Tensor]:
        """选择分层动作"""
        self.steps += 1
        
        # 更新epsilon
        self.epsilon = max(self.epsilon_end, 
                          self.epsilon_start - (self.epsilon_start - self.epsilon_end) * 
                          self.steps / self.epsilon_decay)
        
        with torch.no_grad():
            # 前向传播
            outputs = self.policy_net(
                graph_state['node_features'],
                graph_state['task_adj'],
                graph_state['resource_adj']
            )
            
            high_level_probs = outputs['high_level_probs']
            node_embeddings = outputs['node_embeddings']
            global_features = outputs['global_features']
            
            # 选择高层动作
            if random.random() < self.epsilon:
                high_action = random.randint(0, 3)
            else:
                high_action = torch.argmax(high_level_probs, dim=-1).item()
            
            self.high_action_usage[high_action] += 1
            
            # 获取低层动作参数
            if high_action in [0, 1]:  # 调度或派遣需要节点对
                # 为简化，返回随机参数
                low_action_params = torch.randn(10)  # 假设最多10个候选
            else:
                low_action_params = torch.randn(1)
            
            return int(high_action), low_action_params
    
    def store_transition(self, graph_state, high_action, low_action, reward, next_graph_state, done):
        """存储经验"""
        self.memory.push(graph_state, high_action, low_action, reward, next_graph_state, done)
    
    # ...existing code...

    def update(self):
        """更新网络"""
        if len(self.memory) < self.batch_size:
            return 0
        
        # 采样批次
        batch = self.memory.sample(self.batch_size)
        graph_states, high_actions, low_actions, rewards, next_graph_states, dones = batch
        
        # 将批次数据转换为张量（简化处理）
        rewards = torch.FloatTensor(rewards)
        dones = torch.BoolTensor(dones)
        high_actions = torch.LongTensor(high_actions)
        
        # 计算当前网络输出
        current_outputs = []
        for gs in graph_states:
            output = self.policy_net(gs['node_features'], gs['task_adj'], gs['resource_adj'])
            current_outputs.append(output)
        
        # 计算目标值（简化版本）
        current_values = torch.stack([out['state_value'].squeeze() for out in current_outputs])
        
        with torch.no_grad():
            next_outputs = []
            for gs in next_graph_states:
                output = self.target_net(gs['node_features'], gs['task_adj'], gs['resource_adj'])
                next_outputs.append(output)
            
            next_values = torch.stack([out['state_value'].squeeze() for out in next_outputs])
            target_values = rewards + (self.gamma * next_values * ~dones)
        
        # 计算损失
        value_loss = F.mse_loss(current_values, target_values)
        
        # 策略损失（修复维度问题）
        policy_loss = 0
        for i, output in enumerate(current_outputs):
            high_probs = output['high_level_probs']  # [1, 4]
            advantage = target_values[i] - current_values[i]
            
            # 修复索引问题
            action_idx = high_actions[i].item()
            if action_idx < high_probs.size(-1):  # 确保索引在范围内
                log_prob = torch.log(high_probs[0, action_idx] + 1e-8)  # 添加小值防止log(0)
                policy_loss += -log_prob * advantage.detach()
        
        policy_loss = policy_loss / len(current_outputs)
        
        total_loss = value_loss + policy_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()
        
        # 更新目标网络
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return total_loss.item()

def main():
    """主训练函数"""
    # 配置参数
    config = Config()
    config.num_jobs = 6  # 减少作业数量以适应图网络
    config.num_machines = 3
    config.num_distributors = 2
    config.min_operations = 2
    config.max_operations = 3
    
    print("生成FJSP-DP场景...")
    scenario = FlexibleJobShopScenario(config=config)
    
    print(f"场景信息:")
    print(f"- 作业数量: {len(scenario.jobs)}")
    print(f"- 机器数量: {len(scenario.machines)}")
    print(f"- 配送商数量: {len(scenario.distributors)}")
    
    # 创建环境和智能体
    env = FJSSPGraphEnvironment(scenario)
    node_features = 10  # 节点特征维度
    hidden_dim = 64     # 隐藏层维度
    
    agent = HRLGATAgent(node_features, hidden_dim, config)
    
    # 训练参数
    episodes = 150  # 图网络训练较慢，减少episode数
    stats = defaultdict(list)
    
    print(f"\n开始HRL+GAT训练 {episodes} episodes...")
    start_time = time.time()
    
    for episode in range(episodes):
        graph_state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            # 选择分层动作
            high_action, low_action_params = agent.select_action(graph_state)
            
            # 执行动作
            next_graph_state, reward, done, info = env.step(high_action, low_action_params)
            
            # 存储经验
            agent.store_transition(graph_state, high_action, low_action_params, 
                                 reward, next_graph_state, done)
            
            # 更新网络
            loss = agent.update()
            
            # 更新状态和统计
            graph_state = next_graph_state
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
            print(f"  高层动作使用: {dict(agent.high_action_usage)}")
    
    total_time = time.time() - start_time
    
    # 保存结果
    result_data = {
        'stats': stats,
        'high_action_usage': dict(agent.high_action_usage),
        'config': {
            'episodes': episodes,
            'num_jobs': len(scenario.jobs),
            'num_machines': len(scenario.machines),
            'num_distributors': len(scenario.distributors),
            'hidden_dim': hidden_dim
        }
    }
    
    with open('hrl_gat_results.pkl', 'wb') as f:
        pickle.dump(result_data, f)
    
    # 保存模型
    torch.save(agent.policy_net.state_dict(), 'hrl_gat_model.pth')
    
    print(f"\n训练完成！")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"平均奖励: {np.mean(stats['episode_rewards']):.2f}")
    print(f"平均makespan: {np.mean(stats['makespans']):.2f}")
    print(f"平均完成作业数: {np.mean(stats['completed_jobs']):.2f}")
    print(f"平均派遣作业数: {np.mean(stats['dispatched_jobs']):.2f}")
    print(f"高层动作使用分布: {dict(agent.high_action_usage)}")
    print(f"结果已保存至: hrl_gat_results.pkl")
    print(f"模型已保存至: hrl_gat_model.pth")


def test_trained_model():
    """测试训练好的模型"""
    config = Config()
    config.num_jobs = 4
    config.num_machines = 3
    config.num_distributors = 2
    
    scenario = FlexibleJobShopScenario(config=config)
    env = FJSSPGraphEnvironment(scenario)
    
    node_features = 10
    hidden_dim = 64
    agent = HRLGATAgent(node_features, hidden_dim, config)
    agent.policy_net.load_state_dict(torch.load('hrl_gat_model.pth'))
    agent.epsilon = 0
    
    print("测试训练好的HRL+GAT模型...")
    
    graph_state = env.reset()
    total_reward = 0
    high_action_count = defaultdict(int)
    
    while True:
        high_action, low_action_params = agent.select_action(graph_state)
        high_action_count[high_action] += 1
        
        next_graph_state, reward, done, info = env.step(high_action, low_action_params)
        
        total_reward += reward
        graph_state = next_graph_state
        
        if done:
            break
    
    print(f"测试结果:")
    print(f"  总奖励: {total_reward:.2f}")
    print(f"  Makespan: {env.current_time}")
    print(f"  完成作业: {len(env.completed_jobs)}/{len(env.jobs)}")
    print(f"  派遣作业: {len(env.dispatched_jobs)}/{len(env.jobs)}")
    print(f"  高层动作统计: {dict(high_action_count)}")


if __name__ == "__main__":
    main()
    
    # 取消注释以测试训练好的模型
    # test_trained_model()