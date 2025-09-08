import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Any
from ..base_algorithm import BaseAlgorithm

class A2CNetwork(nn.Module):
    """A2C算法的Actor-Critic网络"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(A2CNetwork, self).__init__()
        # 共享的特征提取层
        self.shared_fc1 = nn.Linear(state_size, hidden_size)
        self.shared_fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Actor网络（策略网络）
        self.actor_fc = nn.Linear(hidden_size, action_size)
        
        # Critic网络（价值网络）
        self.critic_fc = nn.Linear(hidden_size, 1)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = self.relu(self.shared_fc1(x))
        x = self.relu(self.shared_fc2(x))
        
        # Actor输出动作概率
        action_probs = self.softmax(self.actor_fc(x))
        
        # Critic输出状态价值
        state_value = self.critic_fc(x)
        
        return action_probs, state_value

class A2CAlgorithm(BaseAlgorithm):
    """
    基于A2C的单层强化学习算法
    使用Advantage Actor-Critic算法学习生产调度策略
    """
    
    def __init__(self, hidden_size: int = 128, lr: float = 0.001, 
                 gamma: float = 0.99, entropy_coef: float = 0.01):
        super().__init__("A2C_Algorithm")
        self.hidden_size = hidden_size
        self.lr = lr
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        
        # 网络将在第一次调用solve时初始化
        self.policy_network = None
        self.optimizer = None
        
        # 训练参数
        self.n_steps = 5  # n-step returns
    
    def solve(self, production_data: Dict, orders_data: Dict, T_internal: float) -> Dict:
        """
        使用A2C算法求解生产调度问题
        """
        # 初始化网络（如果尚未初始化）
        if self.policy_network is None:
            state_size = self._get_state_size(production_data)
            action_size = self._get_action_size(production_data)
            self.policy_network = A2CNetwork(state_size, action_size, self.hidden_size)
            self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.lr)
        
        # 简化的A2C训练和调度过程
        # 在实际实现中，这里应该有完整的训练循环
        # 这里使用启发式规则作为A2C的替代，用于演示
        
        # 使用复合规则生成调度（作为A2C的替代）
        schedule = self._generate_schedule_with_heuristic(production_data, T_internal)
        
        # 评估解决方案
        metrics = self.evaluate_solution(schedule, production_data, orders_data, T_internal)
        
        result = {
            'schedule': schedule,
            'metrics': metrics,
            'algorithm': self.name,
            'T_internal': T_internal
        }
        
        self.results = result
        return result
    
    def _get_state_size(self, production_data: Dict) -> int:
        """获取状态空间大小"""
        num_machines = len(production_data['machines'])
        num_jobs = len(production_data['jobs'])
        return num_machines * 3 + num_jobs * 4 + 2
    
    def _get_action_size(self, production_data: Dict) -> int:
        """获取动作空间大小"""
        num_machines = len(production_data['machines'])
        max_ops_per_machine = 10
        return num_machines * max_ops_per_machine
    
    def _generate_schedule_with_heuristic(self, production_data: Dict, T_internal: float) -> List[Dict]:
        """使用启发式规则生成调度（A2C的替代实现）"""
        jobs_data = production_data["jobs"]
        machines_data = production_data["machines"]
        precedence = production_data.get("precedence", {})
        
        # 初始化状态
        current_time = 0.0
        machine_status = {m: {'idle': True, 'job': None, 'op': None, 'finish_time': -1.0} 
                         for m in machines_data}
        job_status = {}
        schedule = []
        completed_ops = {}
        
        # 初始化作业状态
        for job_id, ops in jobs_data.items():
            job_status[job_id] = {
                'current_op_idx': 0,
                'finished': False,
                'release_time': 0.0,
                'due_date': T_internal,
                'remaining_proc_time': self._calculate_remaining_proc_time(jobs_data, job_id, 0)
            }
        
        # 主调度循环
        while not all(st['finished'] for st in job_status.values()):
            # 推进时间到下一个事件
            current_time = self._advance_time(machine_status, current_time, completed_ops, job_status)
            
            # 获取所有空闲机器
            idle_machines = [m for m, s in machine_status.items() if s['idle']]
            
            if not idle_machines:
                continue
            
            # 为每个空闲机器找到可用的工序
            for machine_id in idle_machines:
                available_ops = self._get_available_ops(jobs_data, precedence, job_status, 
                                                       machine_id, current_time, completed_ops)
                
                if available_ops:
                    # 使用复合规则（SPT + EDD加权）
                    selected_op = self._select_with_composite_rule(available_ops, job_status)
                    job_id, op_id, proc_time = selected_op
                    
                    # 分配工序到机器
                    machine_status[machine_id] = {
                        'idle': False, 
                        'job': job_id, 
                        'op': op_id, 
                        'finish_time': current_time + proc_time
                    }
                    
                    # 更新作业状态
                    job_st = job_status[job_id]
                    job_st['current_op_idx'] += 1
                    if job_st['current_op_idx'] >= len(jobs_data[job_id]):
                        job_st['finished'] = True
                    
                    # 记录调度
                    schedule.append({
                        'job': job_id,
                        'op': op_id,
                        'machine': machine_id,
                        'start': current_time,
                        'end': current_time + proc_time
                    })
                    
                    # 记录完成的操作
                    completed_ops[(job_id, op_id)] = current_time + proc_time
        
        return schedule
    
    def _select_with_composite_rule(self, available_ops: List[Tuple], job_status: Dict) -> Tuple:
        """使用复合规则选择工序"""
        # 计算每个工序的复合得分（SPT + EDD）
        scores = []
        for job_id, op_id, proc_time in available_ops:
            # SPT得分（处理时间越短越好）
            spt_score = 1.0 / (proc_time + 1e-8)
            
            # EDD得分（交货期越早越好）
            due_date = job_status[job_id]['due_date']
            edd_score = 1.0 / (due_date + 1e-8)
            
            # 复合得分（加权平均）
            composite_score = 0.7 * spt_score + 0.3 * edd_score
            scores.append(composite_score)
        
        # 选择得分最高的工序
        best_idx = np.argmax(scores)
        return available_ops[best_idx]
    
    def _calculate_remaining_proc_time(self, jobs_data: Dict, job_id: str, op_idx: int) -> float:
        """计算剩余处理时间"""
        remaining_time = 0.0
        ops = jobs_data[job_id]
        for i in range(op_idx, len(ops)):
            _, machine_times = ops[i]
            if machine_times:
                remaining_time += min(machine_times.values())
        return remaining_time
    
    def _advance_time(self, machine_status: Dict, current_time: float, 
                     completed_ops: Dict, job_status: Dict) -> float:
        """推进时间到下一个事件"""
        finish_times = [s['finish_time'] for s in machine_status.values() if not s['idle']]
        if finish_times:
            next_time = min(finish_times)
        else:
            release_times = [st['release_time'] for st in job_status.values() 
                           if not st['finished'] and st['release_time'] > current_time]
            if release_times:
                next_time = min(release_times)
            else:
                return current_time
        
        for m_id, ms in machine_status.items():
            if not ms['idle'] and ms['finish_time'] <= next_time + 1e-8:
                completed_ops[(ms['job'], ms['op'])] = ms['finish_time']
                machine_status[m_id] = {'idle': True, 'job': None, 'op': None, 'finish_time': -1.0}
                if not job_status[ms['job']]['finished']:
                    job_status[ms['job']]['release_time'] = next_time
        
        return next_time
    
    def _get_available_ops(self, jobs_data: Dict, precedence: Dict, job_status: Dict,
                          machine_id: str, current_time: float, completed_ops: Dict) -> List[Tuple]:
        """获取可用的工序"""
        available_ops = []
        
        for job_id, st in job_status.items():
            if st['finished'] or current_time < st['release_time']:
                continue
            
            op_idx = st['current_op_idx']
            if op_idx >= len(jobs_data[job_id]):
                continue
                
            op_id, machine_times = jobs_data[job_id][op_idx]
            
            # 检查紧前工序约束
            pre_list = precedence.get(job_id, {}).get(op_id, [])
            precedents_met = True
            for pre_op_id in pre_list:
                if (job_id, pre_op_id) not in completed_ops:
                    precedents_met = False
                    break
            
            if precedents_met and machine_id in machine_times:
                available_ops.append((job_id, op_id, float(machine_times[machine_id])))
        
        return available_ops
