from typing import Any, Dict, List, Tuple, Union
import torch
from dataclasses import dataclass, field
from collections import defaultdict
from entity.config import Config
from entity.dynamic_fjsp_env import EnvironmentState
from entity.job_shop_entities import Job, Machine, DeliveryRequirement, DistributorAssignment, MachineStatus

class FeatureExtractor:
    """特征提取器基类"""
    def __init__(self, config: Config):
        self.config = config
        
    def normalize_time(self, time_value: float) -> float:
        """时间归一化"""
        return time_value / self.config.problem.max_processing_time
    
    def normalize_weight(self, weight: float) -> float:
        """重量归一化"""
        return weight / self.config.problem.max_weight

class JobFeatures(FeatureExtractor):
    """作业特征提取器"""
    def get_basic_features(self, job: Union[Job, Dict]) -> List[float]:
        """获取基础特征"""
        if isinstance(job, Job):
            distributor_id = int(job.distributor_id)  # 确保是整数
            op_count = len(job.operations)
        else:
            distributor_id = int(job.get('distributor_id', 0))  # 确保是整数
            op_count = int(job.get('operation_count', len(job.get('operations', []))))
            
        return [
            self.normalize_time(float(distributor_id)),  # 确保是float
            float(op_count)  # 确保是float
        ]
    
    def get_scheduling_features(self, job: Job) -> List[float]:
        """获取调度相关特征"""
        # 验证并转换基础特征
        basic_features = [float(x) for x in self.get_basic_features(job)]
        
        # 验证并计算当前工序加工时间
        current_op = job.operations[job.current_op]
        processing_time = 0.0
        if current_op.available_machines:
            # 确保processing_times中的值是数值类型
            processing_time = float(current_op.processing_times.get(
                current_op.available_machines[0], 0))
        
        scheduling_features = [
            float(len(job.operations) - job.current_op) / float(len(job.operations)),  # 剩余工序比例
            self.normalize_time(processing_time)  # 当前工序的加工时间            
        ]
        
        # 最终验证所有特征值都是float
        return [float(x) for x in (basic_features + scheduling_features)]
    
    def get_dispatching_features(self, job: Union[Job, Dict], current_time: float) -> List[float]:
        """获取配送相关特征"""
        if isinstance(job, Job):
            # 使用作业的 due_date 或默认值
            completion_time = getattr(job, 'due_date', current_time + self.config.problem.max_processing_time)
            job_info = {
                'completion_time': completion_time,
                'distributor_id': job.distributor_id,
                'tardiness': max(0, current_time - completion_time)
            }
        else:
            job_info = {
                'completion_time': job.get('completion_time', current_time + self.config.problem.max_processing_time),
                'distributor_id': job.get('distributor_id', 0),
                'tardiness': job.get('tardiness', 0)
            }
            
        basic_features = self.get_basic_features(job)
        dispatching_features = [
            self.normalize_time(max(0, job_info['completion_time'] - current_time)),  # 剩余完工时间
            job_info['distributor_id'] / self.config.problem.num_distributors,       # 配送商ID
            self.normalize_time(job_info['tardiness'])                              # 延迟时间
        ]
        return basic_features + dispatching_features

class MachineFeatures(FeatureExtractor):
    """机器特征提取器"""
    def get_features(self, machine: Machine) -> List[float]:
        # 基础特征
        features = [
            machine.utilization,                                   # 利用率
            self.normalize_time(machine.remaining_time),           # 剩余时间
            float(machine.status == MachineStatus.IDLE),           # 是否空闲
            float(machine.status == MachineStatus.BUSY),           # 是否忙碌
            float(machine.status == MachineStatus.SETUP)           # 是否在设置
        ]
        
        # 能力特征 (one-hot编码)
        max_op_types = getattr(self.config.problem, 'max_operation_types', 10)
        capabilities = [0.0] * max_op_types
        for op_type in machine.capabilities:
            if op_type < max_op_types:
                capabilities[op_type] = 1.0
        features.extend(capabilities)
        
        # 时间统计特征
        total_time = max(1e-6, 
            machine.total_busy_time + 
            machine.total_idle_time + 
            machine.total_setup_time)
            
        features.extend([
            machine.total_busy_time / total_time,  # 忙碌时间占比
            machine.total_idle_time / total_time,   # 空闲时间占比
            machine.total_setup_time / total_time   # 设置时间占比
        ])
        
        # 历史作业特征
        features.extend([
            len(machine.job_history),  # 历史作业数量
            sum(job.get('processing_time', 0) for job in machine.job_history) / max(1, len(machine.job_history))  # 平均处理时间
        ])
        
        return features

class BatchFeatures(FeatureExtractor):
    """批次特征提取器"""
    def get_features(self, batch: Dict, current_time: float) -> List[float]:
        current_load = sum(j['weight'] for j in batch['assigned_jobs'])
        return [
            (batch['max_capacity'] - current_load) / batch['max_capacity'],  # 剩余容量比例
            self.normalize_time(batch['earliest_start'] - current_time),     # 最早开始时间
            self.normalize_time(batch['latest_start'] - current_time),       # 最晚开始时间
            len(batch['assigned_jobs']) / self.config.problem.max_jobs_per_batch,# 已分配数量
            batch['distributor_id'] / self.config.problem.num_distributors,  # 配送商ID
            batch.get('utilization', 0)                                      # 当前利用率
        ]

@dataclass
class FeatureCache:
    """特征缓存"""
    features: Dict[str, Dict[int, torch.Tensor]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    ttl: int = 100  # 缓存生存期

class FeatureBuilder(FeatureExtractor):
    """特征构建器"""
    def __init__(self, config: Config):
        super().__init__(config)
        self.job_extractor = JobFeatures(config)
        self.machine_extractor = MachineFeatures(config)
        self.batch_extractor = BatchFeatures(config)
        self.cache = FeatureCache()

    def build_features(self, state: Dict[str, Any], feature_type: str) -> Dict[str, torch.Tensor]:
        """构建指定类型的特征"""
        feature_builders = {
            'meta': self._build_meta_features,
            'scheduling': self._build_scheduling_features,
            'dispatching': self._build_dispatching_features
        }
        
        if feature_type not in feature_builders:
            raise ValueError(f"不支持的特征类型: {feature_type}")
            
        return feature_builders[feature_type](state)

    def _build_meta_features(self, state: EnvironmentState) -> Dict[str, torch.Tensor]:
        """构建元控制器特征
        
        用于决定是否触发新的调度方案
        特征包含:
        1. q_new: 新到达作业比例
        2. sigma_mach_t: 机器负载不平衡程度
        3. urgency_ratio: 紧急程度
        4. schedule_age: 当前调度方案的年龄
        5. system_pressure: 系统压力指标
        """
        current_time = state.current_time
        jobs = state.jobs
        machines = state.machines
        last_schedule_time = getattr(state, 'last_schedule_time', 0)
        
        # 1. 新到达作业比例
        new_jobs = [j for j in jobs if getattr(j, 'arrival_time', None) is not None and j.arrival_time is not None and j.arrival_time > last_schedule_time]
        q_new = len(new_jobs) / max(1, len(jobs))
        
        # 2. 机器负载不平衡度
        remaining_times = [float(m.remaining_time) for m in machines]
        sigma_mach_t = torch.tensor(remaining_times, dtype=torch.float32).std().item() / self.config.problem.max_processing_time
        
        # 3. 紧急程度
        urgent_threshold = self.config.problem.urgent_threshold
        urgency_ratio = 0
        
        # 4. 当前调度方案的年龄
        schedule_age = self.normalize_time(current_time - last_schedule_time)
        
        # 5. 系统压力指标
        # 考虑剩余加工容量与待加工工作量的比例
        
        system_pressure = self._calculate_system_pressure(
        jobs=state.jobs,
        machines=state.machines,
        current_time=state.current_time
        )

        meta_features = torch.tensor([
            q_new,           # 新作业比例
            sigma_mach_t,    # 负载不平衡度
            urgency_ratio,   # 紧急程度
            schedule_age,    # 调度方案年龄
            system_pressure  # 系统压力
        ], dtype=torch.float32)
        
        return {'meta_features': meta_features}

    def _calculate_system_pressure(self, jobs: List[Job], machines: List[Machine], current_time: float) -> float:
        """计算系统压力指标
        
        Args:
            jobs: 当前所有作业列表
            machines: 所有机器列表
            current_time: 当前时间
        
        Returns:
            float: 系统压力指标 [0,1]，值越大表示系统压力越大
        """
        # 1. 计算待加工工作量
        total_remaining_work = 0
        for job in jobs:
            for op in job.operations:
                if op.status != 'waiting':
                    continue
                # 获取该工序在所有可用机器上的平均加工时间
                avg_processing_time = sum(op.processing_times.values()) / len(op.processing_times)
                # 考虑作业优先级
                weighted_processing_time = avg_processing_time 
                total_remaining_work += weighted_processing_time

        # 2. 计算系统加工能力
        total_machine_capacity = 0
        for machine in machines:
            # 计算当前时间窗口内的可用时间
            available_time = self.config.problem.due_window - current_time
            # 减去当前正在处理的工作剩余时间
            if machine.status == 'busy':
                available_time = max(0, available_time - machine.remaining_time)
            # 考虑机器效率
            machine_efficiency = getattr(machine, 'efficiency', 1.0)
            total_machine_capacity += available_time * machine_efficiency

        # 3. 计算压力指标
        if total_machine_capacity <= 0:
            return 1.0  # 如果没有可用产能，返回最大压力
        
        # 计算比值并限制在[0,1]范围内
        pressure = total_remaining_work / total_machine_capacity
        return min(1.0, pressure)

    def _build_scheduling_features(self, state: EnvironmentState) -> Dict[str, torch.Tensor]:
        """
        构建调度决策特征

        返回内容包括：
        - job_features: 作业节点特征 (N_jobs, F_job)
        - machine_features: 机器节点特征 (N_machines, F_machine)
        - job_adj: 作业间邻接矩阵 (N_jobs, N_jobs)
        - machine_adj: 机器间邻接矩阵 (N_machines, N_machines)
        """
        jobs = state.jobs
        machines = state.machines

        # 1. 提取并验证作业特征
        job_feature_list = []
        for job in jobs:
            features = self.job_extractor.get_scheduling_features(job)
            # 确保所有特征都是数值类型
            validated_features = []
            for f in features:
                if isinstance(f, str):
                    try:
                        f = float(f)
                    except ValueError:
                        f = 0.0  # 默认值
                validated_features.append(float(f))
            job_feature_list.append(validated_features)
        
        job_features = torch.tensor(job_feature_list, dtype=torch.float32)

        # 2. 提取并验证机器特征
        machine_feature_list = []
        for machine in machines:
            features = self.machine_extractor.get_features(machine)
            # 确保所有特征都是数值类型
            validated_features = []
            for f in features:
                if isinstance(f, str):
                    try:
                        f = float(f)
                    except ValueError:
                        f = 0.0  # 默认值
                validated_features.append(float(f))
            machine_feature_list.append(validated_features)
        
        machine_features = torch.tensor(machine_feature_list, dtype=torch.float32)

        # 3. 构建邻接矩阵
        job_adj = self._build_job_adjacency(jobs)
        machine_adj = self._build_machine_adjacency(machines)

        return {
            'job_features': job_features,           # 作业特征矩阵
            'machine_features': machine_features,   # 机器特征矩阵
            'job_adj': job_adj,                     # 作业邻接矩阵
            'machine_adj': machine_adj              # 机器邻接矩阵
        }

    def _build_dispatching_features(self, state: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """构建配送决策特征"""
        jobs = state['jobs']
        batches = state['batches']
        current_time = state['current_time']
        
        # 提取节点特征
        job_features = torch.tensor([
            self.job_extractor.get_dispatching_features(job, current_time) 
            for job in jobs
        ], dtype=torch.float32)
        
        batch_features = torch.tensor([
            self.batch_extractor.get_features(batch, current_time) 
            for batch in batches
        ], dtype=torch.float32)
        
        # 构建有效性掩码
        valid_mask = self._build_valid_mask(jobs, batches, current_time)
        
        return {
            'job_features': job_features,
            'batch_features': batch_features,
            'valid_mask': valid_mask
        }

    def _build_job_adjacency(self, jobs: Union[List[Job], List[Dict]]) -> torch.Tensor:
        """构建作业邻接矩阵"""
        n_jobs = len(jobs)
        adj = torch.zeros((n_jobs, n_jobs))
        
        for i, job1 in enumerate(jobs):
            for j, job2 in enumerate(jobs):
                if i != j:
                    # 获取机器兼容性列表
                    if isinstance(job1, Job):
                        # 假设通过 operations 获取机器兼容性
                        compat1 = set()
                        for op in job1.operations:
                            compat1.update(op.available_machines)
                    else:
                        compat1 = set(job1.get('machine_compatibility', []))
                        
                    if isinstance(job2, Job):
                        compat2 = set()
                        for op in job2.operations:
                            compat2.update(op.available_machines)
                    else:
                        compat2 = set(job2.get('machine_compatibility', []))
                    
                    # 计算工序相似度
                    common_machines = compat1 & compat2
                    similarity = len(common_machines) / max(1, len(compat1))
                    adj[i, j] = similarity
                    
        return adj

    def _build_machine_adjacency(self, machines: Union[List[Machine], List[Dict]]) -> torch.Tensor:
        """构建机器邻接矩阵"""
        n_machines = len(machines)
        adj = torch.zeros((n_machines, n_machines))
        
        for i, m1 in enumerate(machines):
            for j, m2 in enumerate(machines):
                if i != j:
                    # 获取利用率
                    util1 = m1.utilization if isinstance(m1, Machine) else m1['utilization']
                    util2 = m2.utilization if isinstance(m2, Machine) else m2['utilization']
                    
                    # 基于负载差异构建连接强度
                    load_diff = abs(util1 - util2)
                    adj[i, j] = 1.0 / (1.0 + load_diff)
                    
        return adj

    def _build_valid_mask(self, jobs: List[Dict], batches: List[Dict], current_time: float) -> torch.Tensor:
        """构建有效分配掩码"""
        mask = torch.ones((len(jobs), len(batches)), dtype=torch.bool)
        
        for j, job in enumerate(jobs):
            for b, batch in enumerate(batches):
                # 检查时间窗口约束
                if job['completion_time'] > batch['latest_start']:
                    mask[j, b] = False
                    continue
                    
                # 检查容量约束
                current_load = sum(j['weight'] for j in batch['assigned_jobs'])
                if current_load + job['weight'] > batch['max_capacity']:
                    mask[j, b] = False
                    continue
                    
                # 检查配送商匹配
                if job['distributor_id'] != batch['distributor_id']:
                    mask[j, b] = False
                    
        return mask
