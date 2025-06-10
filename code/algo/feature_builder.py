from typing import Any, Dict, List, Tuple
import torch
from dataclasses import dataclass,field
from collections import defaultdict

class FeatureExtractor:
    """特征提取器基类"""
    def __init__(self, config: Dict):
        self.config = config
        
    def normalize_time(self, time_value: float) -> float:
        """时间归一化"""
        return time_value / self.config['max_time']
    
    def normalize_weight(self, weight: float) -> float:
        """重量归一化"""
        return weight / self.config['max_weight']

class JobFeatures(FeatureExtractor):
    """作业特征提取器"""
    def get_basic_features(self, job: Dict) -> List[float]:
        """获取基础特征"""
        return [
            self.normalize_time(job['due_date']),         # 截止日期
            job['priority'] / self.config['max_priority'], # 优先级
            self.normalize_weight(job['weight']),         # 重量
            len(job.get('operations', [])) / self.config['max_operations'] # 工序数
        ]
    
    def get_scheduling_features(self, job: Dict) -> List[float]:
        """获取调度相关特征"""
        basic_features = self.get_basic_features(job)
        scheduling_features = [
            len(job['remaining_operations']) / len(job['operations']), # 剩余工序比例
            self.normalize_time(job['processing_time']),              # 加工时间
            job['machine_compatibility'] / self.config['num_machines'] # 机器兼容性
        ]
        return basic_features + scheduling_features
    

    def get_dispatching_features(self, job: Dict, timestamp) -> List[float]:
        """获取配送相关特征"""
        basic_features = self.get_basic_features(job)
        dispatching_features = [
            self.normalize_time(timestamp.time_until(job['completion_time'])),  # 剩余完工时间
            job['distributor_id'] / self.config['num_distributors'],           # 配送商ID
            self.normalize_time(job.get('tardiness', 0))                      # 延迟时间
        ]
        return basic_features + dispatching_features

class MachineFeatures(FeatureExtractor):
    """机器特征提取器"""
    def get_features(self, machine: Dict) -> List[float]:
        return [
            len(machine['queue']) / self.config['max_queue_length'],   # 队列长度
            machine['utilization'],                                     # 利用率
            self.normalize_time(machine['remaining_time']),            # 剩余时间
            machine['status'] == 'idle'                                # 是否空闲
        ]

class BatchFeatures(FeatureExtractor):
    """批次特征提取器"""
    def get_features(self, batch: Dict, current_time: float) -> List[float]:
        current_load = sum(j['weight'] for j in batch['assigned_jobs'])
        return [
            (batch['max_capacity'] - current_load) / batch['max_capacity'], # 剩余容量比例
            self.normalize_time(batch['earliest_start'] - current_time),   # 最早开始时间
            self.normalize_time(batch['latest_start'] - current_time),     # 最晚开始时间
            len(batch['assigned_jobs']) / self.config['max_batch_size'],   # 已分配数量
            batch['distributor_id'] / self.config['num_distributors'],     # 配送商ID
            batch.get('utilization', 0)                                    # 当前利用率
        ]

@dataclass
class FeatureCache:
    """特征缓存"""
    job_features: Dict[int, torch.Tensor] = field(default_factory=dict)
    machine_features: Dict[int, torch.Tensor] = field(default_factory=dict)
    batch_features: Dict[int, torch.Tensor] = field(default_factory=dict)
    ttl: int = 100  # 缓存生存期

class FeatureBuilder:
    """特征构建器"""
    def __init__(self, config: Dict):
        self.config = config
        self.job_extractor = JobFeatures(config)
        self.machine_extractor = MachineFeatures(config)
        self.batch_extractor = BatchFeatures(config)
        self.cache = FeatureCache()


        # ...existing code...
    
    def create_meta_features(
        self,
        state: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """创建元控制器的输入特征
        
        构造5个全局指标:
        1. q_t: 输入缓冲区中的平均作业数量
        2. sigma_mach_t: 各机器剩余加工时间的标准差
        3. s_due_t: 缓冲区作业的平均时间裕度
        4. rho_urg_t: 缓冲区中紧急作业的比例
        5. mu_load_t: 系统中所有作业按订单类型的估计加工负载
        
        Args:
            state: 系统当前状态，包含:
                - jobs: 所有作业信息
                - machines: 所有机器信息
                - current_time: 当前时间
                
        Returns:
            包含5个全局指标的特征张量
        """
        current_time = state['current_time']
        jobs = state['jobs']
        machines = state['machines']
        
        # 1. 计算平均作业数量
        q_t = len(jobs) / self.config['max_jobs']
        
        # 2. 计算机器负载不平衡指标
        remaining_times = [m['remaining_time'] for m in machines]
        sigma_mach_t = torch.tensor(remaining_times).std().item() / self.config['max_time']
        
        # 3. 计算平均时间裕度
        due_times = []
        for job in jobs:
            time_to_due = job['due_date'] - current_time
            due_times.append(max(0, time_to_due))
        s_due_t = (sum(due_times) / len(jobs)) / self.config['max_time'] if jobs else 0
        
        # 4. 计算紧急作业比例
        urgent_threshold = self.config['urgent_time_threshold']
        urgent_jobs = sum(1 for job in jobs if (job['due_date'] - current_time) <= urgent_threshold)
        rho_urg_t = urgent_jobs / len(jobs) if jobs else 0
        
        # 5. 计算估计加工负载
        total_load = 0
        for job in jobs:
            # 考虑剩余工序的加工时间
            remaining_time = sum(op['processing_time'] for op in job['remaining_operations'])
            # 加权考虑作业优先级
            weighted_load = remaining_time * (1 + job['priority'] / self.config['max_priority'])
            total_load += weighted_load
        mu_load_t = total_load / self.config['max_load']
        
        # 构造特征张量
        meta_features = torch.tensor([
            q_t,
            sigma_mach_t,
            s_due_t,
            rho_urg_t,
            mu_load_t
        ], dtype=torch.float32)
        
        return {'meta_features': meta_features}
    
    def _normalize_meta_features(self, features: torch.Tensor) -> torch.Tensor:
        """归一化元控制器特征
        
        使用配置中的最大值进行归一化，确保特征值在[0,1]范围内
        """
        max_values = torch.tensor([
            1.0,                    # q_t 已经归一化
            1.0,                    # sigma_mach_t 已经归一化
            1.0,                    # s_due_t 已经归一化
            1.0,                    # rho_urg_t 是比例值
            1.0                     # mu_load_t 已经归一化
        ], dtype=torch.float32)
        
        return torch.clip(features / max_values, 0, 1)


    def create_dispatching_features(
        self,
        state: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """创建配送决策的输入特征
        
        Args:
            state: 系统当前状态，包含:
                - jobs: 待分配的作业列表
                - batches: 配送批次列表
                - current_time: 当前时间
                
        Returns:
            Dict 包含:
                - job_features: 作业特征矩阵 [n_jobs, feat_dim]
                - batch_features: 批次特征矩阵 [n_batches, feat_dim]
                - valid_mask: 有效分配掩码 [n_jobs, n_batches]
        """
        jobs = state['jobs']
        batches = state['batches']
        current_time = state['current_time']
        
        # 1. 提取作业特征
        job_features = []
        distributor_ids = []
        
        for job in jobs:
            features = self.job_extractor.get_dispatching_features(job, current_time)
            job_features.append(features)
            distributor_ids.append(job['distributor_id'])
        
        # 2. 提取批次特征
        batch_features = []
        for batch in batches:
            features = self.batch_extractor.get_features(batch, current_time)
            batch_features.append(features)
        
        # 3. 创建有效分配掩码
        valid_mask = self.create_valid_mask(jobs, batches, current_time)
        
        return {
            'job_features': torch.tensor(job_features, dtype=torch.float32),
            'batch_features': torch.tensor(batch_features, dtype=torch.float32),
            'distributor_ids': torch.tensor(distributor_ids, dtype=torch.long),
            'valid_mask': valid_mask
        }


    def create_scheduling_features(
        self,
        state: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """创建调度决策的输入特征
        
        Args:
            state: 系统当前状态，包含:
                - jobs: 待调度的作业列表
                - machines: 机器状态列表
                - current_time: 当前时间
                
        Returns:
            Dict 包含:
                - job_features: 作业特征矩阵 [n_jobs, feat_dim]
                - machine_features: 机器特征矩阵 [n_machines, feat_dim]
                - job_adj: 作业邻接矩阵 [n_jobs, n_jobs]
                - machine_adj: 机器邻接矩阵 [n_machines, n_machines]
        """
        jobs = state['jobs']
        machines = state['machines']
        current_time = state['current_time']
        
        # 1. 提取特征
        job_features = []
        machine_features = []
        
        # 作业特征
        for job in jobs:
            features = self.job_extractor.get_scheduling_features(job)
            job_features.append(features)
        
        # 机器特征    
        for machine in machines:
            features = self.machine_extractor.get_features(machine)
            machine_features.append(features)
        
        # 2. 构建邻接矩阵
        n_jobs = len(jobs)
        n_machines = len(machines)
        
        # 作业邻接矩阵：基于工序相似度和时间窗口重叠
        job_adj = torch.zeros((n_jobs, n_jobs))
        for i in range(n_jobs):
            for j in range(n_jobs):
                if i != j:
                    # 计算工序相似度
                    ops_similarity = len(
                        set(jobs[i]['machine_compatibility']) & 
                        set(jobs[j]['machine_compatibility'])
                    ) / len(set(jobs[i]['machine_compatibility']))
                    
                    # 计算时间窗口重叠
                    time_overlap = min(
                        jobs[i]['due_date'], 
                        jobs[j]['due_date']
                    ) - max(
                        current_time + jobs[i]['processing_time'],
                        current_time + jobs[j]['processing_time']
                    )
                    
                    # 综合评分
                    job_adj[i, j] = ops_similarity if time_overlap > 0 else 0
        
        # 机器邻接矩阵：基于工作负载和物理布局
        machine_adj = torch.zeros((n_machines, n_machines))
        for i in range(n_machines):
            for j in range(n_machines):
                if i != j:
                    # 计算负载相似度
                    load_diff = abs(
                        machines[i]['utilization'] - 
                        machines[j]['utilization']
                    )
                    
                    # 考虑物理布局（如果有）
                    layout_dist = machines[i].get('location', {}).get(
                        'distance_to', {}
                    ).get(str(machines[j]['id']), float('inf'))
                    
                    # 综合评分
                    machine_adj[i, j] = 1.0 / (1.0 + load_diff) if layout_dist < float('inf') else 0
        
        return {
            'job_features': torch.tensor(job_features, dtype=torch.float32),
            'machine_features': torch.tensor(machine_features, dtype=torch.float32),
            'job_adj': job_adj,
            'machine_adj': machine_adj
        }


    def create_valid_mask(
        self,
        jobs: List[Dict],
        batches: List[Dict],
        current_time: float
    ) -> torch.Tensor:
        """创建有效分配掩码"""
        n_jobs = len(jobs)
        n_batches = len(batches)
        mask = torch.zeros((n_jobs, n_batches), dtype=torch.bool)
        
        for j, job in enumerate(jobs):
            for b, batch in enumerate(batches):
                if self._check_assignment_validity(job, batch, current_time):
                    mask[j, b] = True
        
        return mask
    
    def _check_assignment_validity(
        self,
        job: Dict,
        batch: Dict,
        current_time: float
    ) -> bool:
        """检查分配是否有效"""
        # 检查配送商匹配
        if job['distributor_id'] != batch['distributor_id']:
            return False
            
        # 检查容量约束
        current_load = sum(j['weight'] for j in batch['assigned_jobs'])
        if current_load + job['weight'] > batch['max_capacity']:
            return False
            
        # 检查时间窗口约束
        n_jobs = len(batch['assigned_jobs']) + 1
        processing_time = self.config['base_processing_time'] + n_jobs * self.config['per_job_time']
        completion_time = current_time + processing_time
        if completion_time > batch['latest_start']:
            return False
            
        return True