"""
算例生成器：基于论文Table 2，生成FJSP-DP工序加工时间与交付要求的标准算例
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import sys
import os
from config import Config
from data_structures import Job, Operation, Distributor, Machine, DeliveryRequirement

@dataclass
class FlexibleJobShopScenario:
    """FJSP-DP算例生成器
    
    使用实体类生成包含加工时间和交付要求的标准算例。
    支持灵活的机器分配和批次调度需求。
    """
    config: Config
    jobs: List[Job]=field(default_factory=list)
    machines: List[Machine] = field(default_factory=list)
    distributors: List[Distributor] = field(default_factory=list)
    
    def __post_init__(self):
        """初始化后自动生成算例"""
        self.num_jobs = self.config.num_initial_jobs
        self.num_machines = self.config.num_machines
        self.num_distributors = self.config.num_distributors
    
        # 生成算例
        self._generate_machines()
        self._generate_jobs()
        self._generate_distributors()
        self._update_job_distributor_mapping()
    
    
    def _generate_machines(self):
        """初始化机器列表"""
        self.machines = [
            Machine(
                machine_id=m,
                current_job=-1, # 代表当前无正在加工Job
                remaining_time=0,
                total_busy_time=0,
                total_idle_time=0,
                processed_jobs=[],
                status="waiting"
            ) for m in range(self.num_machines)
        ]
    
    def _generate_one_job(self, j):
            # 生成数量
            job_amount = np.random.randint(
                self.config.min_job_amount,
                self.config.max_job_amount + 1
            )

            # 生成工序数量
            num_ops = np.random.randint(
                self.config.min_operations,
                self.config.max_operations + 1
            )
            
            # 生成工序
            operations = []
            for o in range(num_ops):
                # 为工序分配机器
                n_machines = np.random.randint(
                    self.config.min_machines_per_op,
                    min(self.config.max_machines_per_op, self.num_machines) + 1
                )
                available_machines = sorted(
                    np.random.choice(range(self.num_machines), n_machines, replace=False)
                )
                
                # 生成加工时间
                processing_times = {
                    m: np.random.randint(
                        self.config.min_processing_time,
                        self.config.max_processing_time + 1
                    )
                    for m in available_machines
                }
                
                # 创建工序对象
                operation = Operation(
                    operation_id=o,
                    job_id=j,
                    available_machine_ids=available_machines,
                    processing_times=processing_times
                )
                operations.append(operation)
                

            job = Job(
                job_id=j,
                amount=job_amount,
                operations=operations,
                # 分配给的配送商ID,注意区间范围
                distributor_id=np.random.randint(0, self.num_distributors)      
            )

            return job
    

    def _generate_jobs(self):
        """生成工件及其工序"""
        for j in range(self.num_jobs):
            job = self._generate_one_job(j)
            self.jobs.append(job)
    
    def _update_job_distributor_mapping(self):
        for job in self.jobs:
            if self.distributors:
                distributor = self.distributors[job.distributor_id]
                if distributor.delivery_requirements:
                    job.due_date = max(distributor.delivery_requirements.due_times)
                    job.earliest_due_date = min(distributor.delivery_requirements.due_times)

    def _generate_distributor_based_due_times(self, distributor_id, n_req, min_time, max_time):
        """基于配送商类型的due_times设置（配送商差异化策略）"""
        
        # 不同配送商有不同的时间偏好
        if distributor_id % 3 == 0:
            # 快速配送商：偏向早期时间窗口
            weights = np.array([3.0, 2.0, 1.0][:n_req])
            distributor_type = "快速配送商"
        elif distributor_id % 3 == 1:
            # 标准配送商：均匀分布
            weights = np.ones(n_req)
            distributor_type = "标准配送商"
        else:
            # 经济配送商：偏向后期时间窗口
            weights = np.array([1.0, 2.0, 3.0][:n_req])
            distributor_type = "经济配送商"
        
        
        # 归一化权重
        weights = weights / weights.sum()
        
        # 根据权重生成累积分布
        cumulative_weights = np.cumsum(weights)
        
        # 生成due_times
        due_times = []
        for i, cum_weight in enumerate(cumulative_weights):
            due_time = min_time + cum_weight * (max_time - min_time)
            due_times.append(int(due_time))
        
        # 确保时间序列单调递增
        due_times = sorted(due_times)
        
        
        return due_times
    
    def _generate_distributors(self):
        """生成配送商分配"""
        for d in range(self.num_distributors):
            assigned_jobs = [j.job_id for j in self.jobs if j.distributor_id == d]
            if not assigned_jobs:
                continue
                
            """为指定配送商生成交付要求"""
            # min_time = self.config.earliest_delivery_time
            # max_time = self.config.latest_delivery_time
            # 估算所有工件的最短和最长加工+配送时间
            min_proc = min([sum([min(op.processing_times.values()) for op in job.operations]) for job in self.jobs])
            max_proc = max([sum([max(op.processing_times.values()) for op in job.operations]) for job in self.jobs])
            delivery_buffer = 10  # 可调节

            min_time = int(min_proc * 0.1)
            max_time = int(max_proc * 1.1 + delivery_buffer)
            n_req = np.random.randint(
                    self.config.min_delivery_requirements,
                    self.config.max_delivery_requirements + 1
            )
            fixed_ratios_list = [
                [0.3, 0.6, 1.0],
                [0.45, 0.8, 1.0],
                [0.2, 0.5, 1.0],
                [0.4, 0.7, 1.0],
                [0.5, 1.0],
                [0.75, 1.0]
            ]
                        
            # 随机选择一种固定比例
            ratios = fixed_ratios_list[np.random.randint(0, len(fixed_ratios_list))]

            
            # 使用配送商差异化策略生成due_times
            due_times = self._generate_distributor_based_due_times(
                d, n_req, min_time, max_time
            )
            
            weights = np.random.uniform(
                self.config.min_load_ratio,
                1.0,
                n_req
            )
            
            delivery_requirements = DeliveryRequirement(
                distributor_id=d,
                due_times=list(due_times),
                ratios=[round(r, 2) for r in ratios],
                weights=weights.tolist()   
            )
            
            total_amount = sum(self.jobs[j].amount for j in assigned_jobs)
            distributor = Distributor(
                distributor_id=d,
                delivery_requirements=delivery_requirements,
                assigned_jobs=assigned_jobs,
                total_amount=total_amount,
                completed_batches={},  # 初始化批次列表
                status='waiting',  # 初始状态为等待
                completed_times={},
                overdue_times={}
            )

            self.distributors.append(distributor)
    
    
# 使用示例
if __name__ == "__main__":
    config = Config()

    
    case = FlexibleJobShopScenario(config=config)
    
    # 验证生成的配送商差异化due_times
    print("配送商差异化策略验证:")
    print("="*50)
    for distributor in case.distributors:
        distributor_type = ["快速配送商", "标准配送商", "经济配送商"][distributor.distributor_id % 3]
        print(f"配送商 {distributor.distributor_id} ({distributor_type}):")
        print(f"  交付时间: {distributor.delivery_requirements.due_times}")
        print(f"  比例: {distributor.delivery_requirements.ratios}")
        print(f"  权重: {[round(w, 3) for w in distributor.delivery_requirements.weights]}")
        print(f"  分配工件数: {len(distributor.assigned_jobs)}")
        print("-" * 30)