"""
算例生成器：基于论文Table 2，生成FJSP-DP工序加工时间与交付要求的标准算例
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import sys
import os
import pandas as pd
import csv
from pathlib import Path

# 添加父目录到sys.path以导入其他模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.config import Config
from model.data_structures import Job, Operation, Distributor, Machine, DeliveryRequirement

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
        max_time += min_time
        for i, cum_weight in enumerate(cumulative_weights):
            due_time = min_time + cum_weight * (max_time - min_time)
            due_times.append(int(due_time))
        
        # 确保时间序列单调递增
        due_times = sorted(due_times)
        
        
        return due_times

    
    def _generate_distributors(self):
        """生成配送商分配"""
        # 首先确保所有配送商都被创建，即使没有作业分配给他们
        for d in range(self.num_distributors):
            assigned_jobs = [j for j in self.jobs if j.distributor_id == d]
            
            # 估算所有工件的最短和最长加工+配送时间
            min_proc = float('inf')
            max_proc = float('-inf')
            
            if assigned_jobs:
                for job in assigned_jobs:
                    min_proc = min(min_proc, sum([min(op.processing_times.values()) for op in job.operations]))
                    max_proc = max(max_proc, sum([max(op.processing_times.values()) for op in job.operations]))
            else:
                # 如果没有作业分配给这个配送商，使用默认值
                min_proc = self.config.min_processing_time * self.config.min_operations
                max_proc = self.config.max_processing_time * self.config.max_operations

            # 交付要求数量n_req从U[2,3]中采样
            n_req = np.random.randint(2, 4)  # U[2,3] 均匀分布
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
                d, n_req, min_proc, max_proc
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
            
            total_amount = sum(j.amount for j in assigned_jobs)
            distributor = Distributor(
                distributor_id=d,
                delivery_requirements=delivery_requirements,
                assigned_jobs=[j.job_id for j in assigned_jobs],
                total_amount=total_amount,
                completed_batches={},  # 初始化批次列表
                status='waiting',  # 初始状态为等待
                completed_times={},
                overdue_times={}
            )

            self.distributors.append(distributor)
    
    def export_to_csv(self, output_dir: str = "instances/csv_data", instance_name: Optional[str] = None):
        """
        将算例导出为CSV格式文件
        
        Args:
            output_dir: 输出目录路径
            instance_name: 算例名称，如果为None则自动生成
        """
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 生成算例名称
        if instance_name is None:
            if self.num_jobs <= 30:
                size = "Small"
            elif self.num_jobs <= 100:
                size = "Medium"
            else:
                size = "Large"
            instance_name = f"{size}-{self.num_jobs}J{self.num_machines}M{self.num_distributors}D"
        
        # 导出工件信息
        self._export_jobs_csv(output_path, instance_name)
        
        # 导出工序信息  
        self._export_operations_csv(output_path, instance_name)
        
        # 导出机器信息
        self._export_machines_csv(output_path, instance_name)
        
        # 导出配送商信息
        self._export_distributors_csv(output_path, instance_name)
        
        # 导出算例配置信息
        self._export_instance_config_csv(output_path, instance_name)
        
        print(f"算例 {instance_name} 已导出到 {output_path}")
    
    def _export_jobs_csv(self, output_path: Path, instance_name: str):
        """导出工件信息到CSV"""
        jobs_data = []
        for job in self.jobs:
            jobs_data.append({
                'job_id': job.job_id,
                'amount': job.amount,
                'distributor_id': job.distributor_id,
                'num_operations': len(job.operations),
                'due_date': job.due_date,
                'earliest_due_date': job.earliest_due_date,
                'job_type': job.job_type,
                'arrival_time': job.arrival_time
            })
        
        df = pd.DataFrame(jobs_data)
        df.to_csv(output_path / f"{instance_name}_jobs.csv", index=False)
    
    def _export_operations_csv(self, output_path: Path, instance_name: str):
        """导出工序信息到CSV"""
        operations_data = []
        for job in self.jobs:
            for op in job.operations:
                # 将处理时间字典转换为字符串格式
                processing_times_str = ";".join([f"{m}:{t}" for m, t in op.processing_times.items()])
                available_machines_str = ";".join(map(str, op.available_machine_ids))
                
                operations_data.append({
                    'job_id': op.job_id,
                    'operation_id': op.operation_id,
                    'available_machines': available_machines_str,
                    'processing_times': processing_times_str,
                    'status': op.status
                })
        
        df = pd.DataFrame(operations_data)
        df.to_csv(output_path / f"{instance_name}_operations.csv", index=False)
    
    def _export_machines_csv(self, output_path: Path, instance_name: str):
        """导出机器信息到CSV"""
        machines_data = []
        for machine in self.machines:
            machines_data.append({
                'machine_id': machine.machine_id,
                'status': machine.status,
                'current_job': machine.current_job,
                'remaining_time': machine.remaining_time,
                'total_busy_time': machine.total_busy_time,
                'total_idle_time': machine.total_idle_time
            })
        
        df = pd.DataFrame(machines_data)
        df.to_csv(output_path / f"{instance_name}_machines.csv", index=False)
    
    def _export_distributors_csv(self, output_path: Path, instance_name: str):
        """导出配送商信息到CSV"""
        distributors_data = []
        for distributor in self.distributors:
            # 将列表转换为字符串格式
            due_times_str = ";".join(map(str, distributor.delivery_requirements.due_times))
            ratios_str = ";".join(map(str, distributor.delivery_requirements.ratios))
            weights_str = ";".join([f"{w:.3f}" for w in distributor.delivery_requirements.weights])
            assigned_jobs_str = ";".join(map(str, distributor.assigned_jobs))
            
            distributors_data.append({
                'distributor_id': distributor.distributor_id,
                'due_times': due_times_str,
                'ratios': ratios_str,
                'weights': weights_str,
                'assigned_jobs': assigned_jobs_str,
                'total_amount': distributor.total_amount,
                'status': distributor.status
            })
        
        df = pd.DataFrame(distributors_data)
        df.to_csv(output_path / f"{instance_name}_distributors.csv", index=False)
    
    def _export_instance_config_csv(self, output_path: Path, instance_name: str):
        """导出算例配置信息到CSV"""
        config_data = [{
            'instance_name': instance_name,
            'num_jobs': self.num_jobs,
            'num_machines': self.num_machines,
            'num_distributors': self.num_distributors,
            'min_operations': self.config.min_operations,
            'max_operations': self.config.max_operations,
            'min_processing_time': self.config.min_processing_time,
            'max_processing_time': self.config.max_processing_time,
            'min_job_amount': self.config.min_job_amount,
            'max_job_amount': self.config.max_job_amount,
            'min_delivery_requirements': self.config.min_delivery_requirements,
            'max_delivery_requirements': self.config.max_delivery_requirements
        }]
        
        df = pd.DataFrame(config_data)
        df.to_csv(output_path / f"{instance_name}_config.csv", index=False)

    @classmethod
    def generate_paper_instances_csv(cls, output_dir: str = "instances/csv_data"):
        """
        根据论文参数设置批量生成CSV格式的标准算例
        
        Args:
            output_dir: 输出目录路径
        """
        # 基于论文Table 4的参数设置 - 完整版本
        paper_configs = [
            # 小规模算例 (40-100作业): 初始作业20, 机器10
            (20, 10, 20, 5),   # 动态作业20, 配送商5
            (20, 10, 20, 10),  # 动态作业20, 配送商10
            (20, 10, 20, 15),  # 动态作业20, 配送商15
            (20, 10, 40, 5),   # 动态作业40, 配送商5
            (20, 10, 40, 10),  # 动态作业40, 配送商10
            (20, 10, 40, 15),  # 动态作业40, 配送商15
            (20, 10, 60, 5),   # 动态作业60, 配送商5
            (20, 10, 60, 10),  # 动态作业60, 配送商10
            (20, 10, 60, 15),  # 动态作业60, 配送商15
            (20, 10, 80, 5),   # 动态作业80, 配送商5
            (20, 10, 80, 10),  # 动态作业80, 配送商10
            (20, 10, 80, 15),  # 动态作业80, 配送商15
            
            # 中规模算例 (100-160作业): 初始作业80, 机器20
            (80, 20, 20, 5),   # 动态作业20, 配送商5
            (80, 20, 20, 10),  # 动态作业20, 配送商10
            (80, 20, 20, 15),  # 动态作业20, 配送商15
            (80, 20, 40, 5),   # 动态作业40, 配送商5
            (80, 20, 40, 10),  # 动态作业40, 配送商10
            (80, 20, 40, 15),  # 动态作业40, 配送商15
            (80, 20, 60, 5),   # 动态作业60, 配送商5
            (80, 20, 60, 10),  # 动态作业60, 配送商10
            (80, 20, 60, 15),  # 动态作业60, 配送商15
            (80, 20, 80, 5),   # 动态作业80, 配送商5
            (80, 20, 80, 10),  # 动态作业80, 配送商10
            (80, 20, 80, 15),  # 动态作业80, 配送商15
            
            # 大规模算例 (160-220作业): 初始作业140, 机器30
            (140, 30, 20, 5),  # 动态作业20, 配送商5
            (140, 30, 20, 10), # 动态作业20, 配送商10
            (140, 30, 20, 15), # 动态作业20, 配送商15
            (140, 30, 40, 5),  # 动态作业40, 配送商5
            (140, 30, 40, 10), # 动态作业40, 配送商10
            (140, 30, 40, 15), # 动态作业40, 配送商15
            (140, 30, 60, 5),  # 动态作业60, 配送商5
            (140, 30, 60, 10), # 动态作业60, 配送商10
            (140, 30, 60, 15), # 动态作业60, 配送商15
            (140, 30, 80, 5),  # 动态作业80, 配送商5
            (140, 30, 80, 10), # 动态作业80, 配送商10
            (140, 30, 80, 15), # 动态作业80, 配送商15
        ]
        
        print(f"开始生成 {len(paper_configs)} 个论文标准算例...")
        
        for i, (initial_jobs, machines, dynamic_jobs, distributors) in enumerate(paper_configs, 1):
            # 创建配置
            config = Config()
            config.num_initial_jobs = initial_jobs
            config.num_machines = machines
            config.num_dynamic_jobs = dynamic_jobs
            config.num_distributors = distributors
            
            # 论文Table 2中的参数
            config.min_operations = 1
            config.max_operations = 3
            config.min_processing_time = 5
            config.max_processing_time = 15
            config.min_job_amount = 50
            config.max_job_amount = 100
            
            # 生成算例名称
            total_jobs = initial_jobs + dynamic_jobs
            if total_jobs <= 100:
                size = "Small"
            elif total_jobs <= 160:
                size = "Medium"
            else:
                size = "Large"
            instance_name = f"{size}-{total_jobs}J{machines}M{distributors}D"
            
            print(f"正在生成算例 {i}/{len(paper_configs)}: {instance_name}")
            
            # 生成算例
            scenario = cls(config=config)
            
            # 导出为CSV
            scenario.export_to_csv(output_dir, instance_name)
        
        print(f"所有算例已生成完成，保存在 {output_dir} 目录中")
    
    
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
    
    # 导出单个算例为CSV
    print("\n导出当前算例为CSV格式...")
    case.export_to_csv()
    
    # 生成所有论文标准算例的CSV文件
    print("\n生成所有论文标准算例的CSV文件...")
    FlexibleJobShopScenario.generate_paper_instances_csv()
